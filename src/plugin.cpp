#include <algorithm>
#include <clap/clap.h>
#include <cmath>
#include <cstdio>
#include <cstring>

// WSOLA tuned for 256 frames / 48 kHz.
// WIN = 2×HOP → 50% overlap; Hann window satisfies COLA at this ratio.
static constexpr int HOP     = 256;
static constexpr int WIN     = 512;
static constexpr int RING    = 1 << 15;  // 32768 samples ≈ 0.68 s @ 48 kHz
static constexpr int RING_MSK = RING - 1;
static constexpr int MIN_GAP = WIN + HOP; // absolute minimum read-behind-write distance

// Tuneable defaults (also the CLAP param defaults)
static constexpr int DEF_SEARCH      = 64;    // cross-correlation search range (samples)
static constexpr int DEF_JUMP_BACK   = 512;   // nominal jump size on gap underflow (samples)
static constexpr int DEF_INITIAL_GAP = 2048;  // initial/target rd-behind-wr gap (samples) = latency

static float s_hann[WIN];
static void  build_hann() {
    for (int i = 0; i < WIN; ++i)
        s_hann[i] = 0.5f * (1.f - cosf(2.f * M_PI * i / WIN));
}

// ── Per-channel WSOLA state ───────────────────────────────────────────────────

struct Chan {
    float   ring[RING] = {};
    float   olap[WIN]  = {};
    float   prev[WIN]  = {};
    int64_t wr   = 0;
    double  rd   = 0.0;
    bool    warm = false;

    void reset(int initial_gap) {
        std::memset(ring, 0, sizeof ring);
        std::memset(olap, 0, sizeof olap);
        std::memset(prev, 0, sizeof prev);
        wr   = initial_gap;
        rd   = 0.0;
        warm = false;
    }

    void push(const float* in, int n) {
        for (int i = 0; i < n; ++i)
            ring[(wr + i) & RING_MSK] = in[i];
        wr += n;
    }

    // Normalized cross-correlation of prev vs ring[base+d .. +WIN) for d in [-range, range].
    // Returns absolute position of the best match.
    int64_t best_match(int64_t base, int range, float prev_energy) const {
        int64_t best_pos = base;
        float   best_c   = -2.f;
        for (int d = -range; d <= range; ++d) {
            float c = 0.f, energy = 0.f;
            for (int i = 0; i < WIN; ++i) {
                const float s = ring[(base + d + i) & RING_MSK];
                c      += prev[i] * s;
                energy += s * s;
            }
            const float nc = c / sqrtf(prev_energy * energy + 1e-10f);
            if (nc > best_c) { best_c = nc; best_pos = base + d; }
        }
        return best_pos;
    }

    void step(float* out, double ratio, int search, int jump_back, int initial_gap) {
        const int64_t gap = wr - static_cast<int64_t>(rd);
        const int     max_gap = RING - MIN_GAP - search;

        float prev_energy = 0.f;
        if (warm)
            for (int i = 0; i < WIN; ++i) prev_energy += prev[i] * prev[i];

        // ── Gap management ────────────────────────────────────────────────────
        int64_t gp;
        if (gap < MIN_GAP + search) {
            // Pitch up: rd caught up to wr — jump back by jump_back, find best splice
            const int64_t nominal = static_cast<int64_t>(rd) - jump_back;
            gp = warm ? best_match(nominal, search, prev_energy) : nominal;
            rd = static_cast<double>(gp);
        } else if (gap > max_gap) {
            // Pitch down: wr ran ahead of rd — skip rd forward to restore target gap
            const int64_t nominal = static_cast<int64_t>(wr) - initial_gap;
            gp = warm ? best_match(nominal, search, prev_energy) : nominal;
            rd = static_cast<double>(gp);
        } else {
            // Normal: WSOLA search around nominal rd
            gp = warm ? best_match(static_cast<int64_t>(rd), search, prev_energy)
                      : static_cast<int64_t>(rd);
        }

        // ── Extract windowed grain ────────────────────────────────────────────
        float grain[WIN];
        for (int i = 0; i < WIN; ++i)
            grain[i] = ring[(gp + i) & RING_MSK] * s_hann[i];

        // ── Overlap-add → output → shift ──────────────────────────────────────
        for (int i = 0; i < WIN; ++i) olap[i] += grain[i];
        std::memcpy(out, olap, HOP * sizeof(float));
        std::memmove(olap, olap + HOP, (WIN - HOP) * sizeof(float));
        std::memset(olap + WIN - HOP, 0, HOP * sizeof(float));

        std::memcpy(prev, grain, WIN * sizeof(float));
        warm  = true;
        rd   += HOP * ratio;
    }
};

// ── Parameters ────────────────────────────────────────────────────────────────

enum : clap_id {
    PARAM_SEMITONES = 0,
    PARAM_SEARCH,       // WSOLA search range (samples)
    PARAM_JUMP_BACK,    // jump-back size on gap underflow (samples)
    PARAM_INITIAL_GAP,  // initial/target gap = reported latency (samples)
    PARAM_COUNT,
};

static const clap_param_info_t k_params[PARAM_COUNT] = {
    {
        .id = PARAM_SEMITONES, .flags = CLAP_PARAM_IS_AUTOMATABLE | CLAP_PARAM_IS_MODULATABLE,
        .cookie = nullptr, .name = "Pitch", .module = "",
        .min_value = -24.0, .max_value = 24.0, .default_value = 0.0,
    },
    {
        .id = PARAM_SEARCH, .flags = 0,
        .cookie = nullptr, .name = "Search Range", .module = "WSOLA",
        .min_value = 8.0, .max_value = 256.0, .default_value = DEF_SEARCH,
    },
    {
        .id = PARAM_JUMP_BACK, .flags = 0,
        .cookie = nullptr, .name = "Jump Back", .module = "WSOLA",
        .min_value = static_cast<double>(HOP), .max_value = static_cast<double>(HOP * 8),
        .default_value = DEF_JUMP_BACK,
    },
    {
        .id = PARAM_INITIAL_GAP, .flags = 0,
        .cookie = nullptr, .name = "Latency / Gap", .module = "WSOLA",
        .min_value = static_cast<double>(WIN + HOP), .max_value = static_cast<double>(RING / 4),
        .default_value = DEF_INITIAL_GAP,
    },
};

// ── Plugin struct ─────────────────────────────────────────────────────────────

struct PitchShifter {
    clap_plugin_t      plugin;
    const clap_host_t* host;
    double semitones   = 0.0;
    int    search      = DEF_SEARCH;
    int    jump_back   = DEF_JUMP_BACK;
    int    initial_gap = DEF_INITIAL_GAP;
    Chan   ch[2];
};

static PitchShifter* self(const clap_plugin_t* p) {
    return static_cast<PitchShifter*>(p->plugin_data);
}

static bool plugin_init    (const clap_plugin_t*)         { return true; }
static void plugin_destroy (const clap_plugin_t* p)       { delete self(p); }
static void plugin_deactivate    (const clap_plugin_t*)   {}
static bool plugin_start_processing(const clap_plugin_t*) { return true; }
static void plugin_stop_processing (const clap_plugin_t*) {}
static void plugin_on_main_thread  (const clap_plugin_t*) {}

static bool plugin_activate(const clap_plugin_t* p, double, uint32_t, uint32_t) {
    auto* s = self(p);
    for (auto& c : s->ch) c.reset(s->initial_gap);
    return true;
}

static void plugin_reset(const clap_plugin_t* p) {
    auto* s = self(p);
    for (auto& c : s->ch) c.reset(s->initial_gap);
}

static void apply_params(PitchShifter* s, const clap_input_events_t* ev) {
    for (uint32_t i = 0; i < ev->size(ev); ++i) {
        const auto* hdr = ev->get(ev, i);
        if (hdr->type != CLAP_EVENT_PARAM_VALUE) continue;
        const auto* e = reinterpret_cast<const clap_event_param_value_t*>(hdr);
        switch (e->param_id) {
            case PARAM_SEMITONES:   s->semitones   = e->value; break;
            case PARAM_SEARCH:      s->search      = static_cast<int>(e->value); break;
            case PARAM_JUMP_BACK:   s->jump_back   = static_cast<int>(e->value); break;
            case PARAM_INITIAL_GAP: s->initial_gap = static_cast<int>(e->value); break;
        }
    }
}

static clap_process_status plugin_process(const clap_plugin_t* p,
                                           const clap_process_t* proc) {
    auto* s = self(p);
    apply_params(s, proc->in_events);

    if (!proc->audio_inputs_count || !proc->audio_outputs_count)
        return CLAP_PROCESS_CONTINUE;

    const auto& ain  = proc->audio_inputs[0];
    const auto& aout = proc->audio_outputs[0];
    const uint32_t nch   = std::min({ain.channel_count, aout.channel_count, 2u});
    const uint32_t nfr   = proc->frames_count;
    const double   ratio = std::pow(2.0, s->semitones / 12.0);

    for (uint32_t c = 0; c < nch; ++c) {
        s->ch[c].push(ain.data32[c], static_cast<int>(nfr));
        if (nfr >= HOP) {
            s->ch[c].step(aout.data32[c], ratio, s->search, s->jump_back, s->initial_gap);
            if (nfr > HOP)
                std::memset(aout.data32[c] + HOP, 0, (nfr - HOP) * sizeof(float));
        } else {
            std::memset(aout.data32[c], 0, nfr * sizeof(float));
        }
    }
    return CLAP_PROCESS_CONTINUE;
}

static const void* plugin_get_extension(const clap_plugin_t*, const char* id);

// ── Audio ports extension ─────────────────────────────────────────────────────

static uint32_t ap_count(const clap_plugin_t*, bool) { return 1; }
static bool ap_get(const clap_plugin_t*, uint32_t idx, bool,
                   clap_audio_port_info_t* info) {
    if (idx) return false;
    info->id = 0; info->channel_count = 2;
    info->flags = CLAP_AUDIO_PORT_IS_MAIN;
    info->port_type = CLAP_PORT_STEREO;
    info->in_place_pair = CLAP_INVALID_ID;
    std::strncpy(info->name, "Main", sizeof(info->name));
    return true;
}
static const clap_plugin_audio_ports_t s_audio_ports = { ap_count, ap_get };

// ── Latency extension ─────────────────────────────────────────────────────────

static uint32_t latency_get(const clap_plugin_t* p) {
    return static_cast<uint32_t>(self(p)->initial_gap);
}
static const clap_plugin_latency_t s_latency = { latency_get };

// ── Params extension ──────────────────────────────────────────────────────────

static uint32_t par_count(const clap_plugin_t*) { return PARAM_COUNT; }
static bool par_info(const clap_plugin_t*, uint32_t i, clap_param_info_t* o) {
    if (i >= PARAM_COUNT) return false;
    *o = k_params[i]; return true;
}
static bool par_get(const clap_plugin_t* p, clap_id id, double* v) {
    auto* s = self(p);
    switch (id) {
        case PARAM_SEMITONES:   *v = s->semitones;             return true;
        case PARAM_SEARCH:      *v = static_cast<double>(s->search);      return true;
        case PARAM_JUMP_BACK:   *v = static_cast<double>(s->jump_back);   return true;
        case PARAM_INITIAL_GAP: *v = static_cast<double>(s->initial_gap); return true;
    }
    return false;
}
static bool par_to_text(const clap_plugin_t*, clap_id id, double v,
                         char* buf, uint32_t sz) {
    switch (id) {
        case PARAM_SEMITONES:   snprintf(buf, sz, "%.2f st", v);         return true;
        case PARAM_SEARCH:      snprintf(buf, sz, "%d smp", (int)v);     return true;
        case PARAM_JUMP_BACK:   snprintf(buf, sz, "%d smp", (int)v);     return true;
        case PARAM_INITIAL_GAP: snprintf(buf, sz, "%d smp", (int)v);     return true;
    }
    return false;
}
static bool par_from_text(const clap_plugin_t*, clap_id, const char*, double*) {
    return false;
}
static void par_flush(const clap_plugin_t* p,
                      const clap_input_events_t* in,
                      const clap_output_events_t*) {
    apply_params(self(p), in);
}
static const clap_plugin_params_t s_params = {
    par_count, par_info, par_get, par_to_text, par_from_text, par_flush
};

static const void* plugin_get_extension(const clap_plugin_t*, const char* id) {
    if (!std::strcmp(id, CLAP_EXT_AUDIO_PORTS)) return &s_audio_ports;
    if (!std::strcmp(id, CLAP_EXT_LATENCY))     return &s_latency;
    if (!std::strcmp(id, CLAP_EXT_PARAMS))      return &s_params;
    return nullptr;
}

// ── Factory ───────────────────────────────────────────────────────────────────

static const clap_plugin_descriptor_t k_desc = {
    .clap_version = CLAP_VERSION_INIT,
    .id           = "com.example.pitch-shifter",
    .name         = "Pitch Shifter",
    .vendor       = "Example",
    .url          = "", .manual_url = "", .support_url = "",
    .version      = "0.1.0",
    .description  = "WSOLA pitch shifter — 256 / 48 kHz",
    .features     = (const char*[]){
        CLAP_PLUGIN_FEATURE_AUDIO_EFFECT,
        CLAP_PLUGIN_FEATURE_PITCH_SHIFTER,
        nullptr
    },
};

static uint32_t fac_count(const clap_plugin_factory_t*) { return 1; }
static const clap_plugin_descriptor_t* fac_desc(const clap_plugin_factory_t*, uint32_t i) {
    return i ? nullptr : &k_desc;
}
static const clap_plugin_t* fac_create(const clap_plugin_factory_t*,
                                        const clap_host_t* host, const char* id) {
    if (std::strcmp(id, k_desc.id)) return nullptr;
    auto* s   = new PitchShifter{};
    s->host   = host;
    s->plugin = {
        .desc             = &k_desc,
        .plugin_data      = s,
        .init             = plugin_init,
        .destroy          = plugin_destroy,
        .activate         = plugin_activate,
        .deactivate       = plugin_deactivate,
        .start_processing = plugin_start_processing,
        .stop_processing  = plugin_stop_processing,
        .reset            = plugin_reset,
        .process          = plugin_process,
        .get_extension    = plugin_get_extension,
        .on_main_thread   = plugin_on_main_thread,
    };
    return &s->plugin;
}
static const clap_plugin_factory_t s_factory = { fac_count, fac_desc, fac_create };

// ── Entry point ───────────────────────────────────────────────────────────────

static bool entry_init(const char*) { build_hann(); return true; }
static void entry_deinit() {}
static const void* entry_factory(const char* id) {
    return std::strcmp(id, CLAP_PLUGIN_FACTORY_ID) ? nullptr : &s_factory;
}

CLAP_EXPORT const clap_plugin_entry_t clap_entry = {
    .clap_version = CLAP_VERSION_INIT,
    .init         = entry_init,
    .deinit       = entry_deinit,
    .get_factory  = entry_factory,
};
