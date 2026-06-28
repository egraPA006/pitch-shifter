#include <algorithm>
#include <clap/clap.h>
#include <cmath>
#include <cstdio>
#include <cstring>

// OLA time-stretch + 4-point cubic-Hermite resampler pitch shifter.
//
// Pipeline: input → OLA (H_a = HOP/ratio, no grain search)
//                 → cubic Hermite resampler (×ratio) → output
//
// Why no grain search: WSOLA's cross-correlation finds grains that look like
// the previous one, effectively zeroing out small pitch changes.  Plain OLA
// with a fractional hop is what actually shifts pitch for all ratio values.
// Latency = initial_gap samples (constant; DAW PDC compensates).

static constexpr int WIN      = 256;
static constexpr int HOP      = 128;    // synthesis hop; WIN=2×HOP → 50 % overlap, COLA sum = 1
static constexpr int RING     = 1 << 15;
static constexpr int RING_MSK = RING - 1;
static constexpr int MIN_GAP  = WIN + HOP;
static constexpr int MAX_GAP  = RING - MIN_GAP;
static constexpr int IBUF     = 4096;   // intermediate ring (OLA output)

static constexpr int DEF_INITIAL_GAP = 512;  // ≈ 10.7 ms

static float s_hann[WIN];
static void  build_hann() {
    for (int i = 0; i < WIN; ++i)
        s_hann[i] = 0.5f * (1.f - cosf(2.f * M_PI * i / WIN));
}

// 4-point cubic Hermite interpolation, t ∈ [0, 1).
// Much better high-frequency response than linear; near-alias-free for ratio ≤ 2.
static inline float hermite4(float pm1, float p0, float p1, float p2, float t) {
    const float a = -0.5f*pm1 + 1.5f*p0 - 1.5f*p1 + 0.5f*p2;
    const float b =       pm1 - 2.5f*p0 + 2.0f*p1 - 0.5f*p2;
    const float c = -0.5f*pm1            + 0.5f*p1;
    return ((a*t + b)*t + c)*t + p0;
}

// ── Per-channel state ─────────────────────────────────────────────────────────

struct Chan {
    float   ring[RING] = {};
    int64_t wr   = 0;
    double  rd   = 0.0;     // analysis read head; advances by HOP/ratio per step

    float  olap[WIN] = {};  // OLA accumulator (50 % overlap, COLA → no normalization)

    // Intermediate buffer (OLA output, consumed by resampler)
    // +3 headroom: cubic interp reads ibuf[pos-1..pos+2]
    float  ibuf[IBUF + 3] = {};
    int    ibuf_fill = 0;
    double iread     = 0.0;
    float  ibuf_pm1  = 0.f; // sample just before ibuf[0] for the cubic interp boundary

    void reset(int initial_gap) {
        std::memset(ring, 0, sizeof ring);
        std::memset(olap, 0, sizeof olap);
        std::memset(ibuf, 0, sizeof ibuf);
        wr = initial_gap; rd = 0.0;
        ibuf_fill = 0; iread = 0.0; ibuf_pm1 = 0.f;
    }

    void push(const float* in, int n) {
        for (int i = 0; i < n; ++i)
            ring[(wr + i) & RING_MSK] = in[i];
        wr += n;
    }

    // One OLA step.  Reads H_a = HOP/ratio input samples, writes HOP to ibuf.
    // gp = truncate(rd): no cross-correlation search, so small ratios produce
    // a real fractional-hop time-stretch instead of being cancelled out.
    void ola_step(double ratio, int initial_gap) {
        const int64_t gap = wr - static_cast<int64_t>(rd);
        if (gap < MIN_GAP || gap > MAX_GAP)
            rd = static_cast<double>(wr - initial_gap);  // reanchor when out of bounds

        const int64_t gp = static_cast<int64_t>(rd);
        float grain[WIN];
        for (int i = 0; i < WIN; ++i)
            grain[i] = ring[(gp + i) & RING_MSK] * s_hann[i];

        for (int i = 0; i < WIN; ++i) olap[i] += grain[i];

        const int to_write = std::min(HOP, IBUF - ibuf_fill);
        std::memcpy(ibuf + ibuf_fill, olap, to_write * sizeof(float));
        ibuf_fill += to_write;

        std::memmove(olap, olap + HOP, (WIN - HOP) * sizeof(float));
        std::memset(olap + WIN - HOP, 0, HOP * sizeof(float));

        rd += HOP / ratio;
    }

    void process(const float* in, float* out, int n, double ratio, int initial_gap) {
        push(in, n);

        // Compact: evict samples already consumed, save the last one for interp boundary
        const int consumed = static_cast<int>(iread);
        if (consumed > 0 && consumed <= ibuf_fill) {
            ibuf_pm1   = ibuf[consumed - 1];
            ibuf_fill -= consumed;
            std::memmove(ibuf, ibuf + consumed, (ibuf_fill + 3) * sizeof(float));
            iread -= consumed;
        }

        // Fill intermediate until resampler can produce n output samples
        const int needed = static_cast<int>(n * ratio) + 3;
        while (ibuf_fill < needed)
            ola_step(ratio, initial_gap);

        // Cubic Hermite resampler: advances iread by ratio per output sample
        for (int i = 0; i < n; ++i) {
            const int   pos  = static_cast<int>(iread);
            const float frac = static_cast<float>(iread - pos);
            const float pm1  = pos > 0 ? ibuf[pos - 1] : ibuf_pm1;
            out[i] = hermite4(pm1, ibuf[pos], ibuf[pos + 1], ibuf[pos + 2], frac);
            iread += ratio;
        }
    }
};

// ── Parameters ────────────────────────────────────────────────────────────────

enum : clap_id { PARAM_SEMITONES = 0, PARAM_INITIAL_GAP, PARAM_COUNT };

static const clap_param_info_t k_params[PARAM_COUNT] = {
    {
        .id = PARAM_SEMITONES,
        .flags = CLAP_PARAM_IS_AUTOMATABLE | CLAP_PARAM_IS_MODULATABLE,
        .cookie = nullptr, .name = "Pitch", .module = "",
        .min_value = -24.0, .max_value = 24.0, .default_value = 0.0,
    },
    {
        .id = PARAM_INITIAL_GAP, .flags = 0,
        .cookie = nullptr, .name = "Latency / Gap", .module = "",
        .min_value = static_cast<double>(MIN_GAP),
        .max_value = static_cast<double>(RING / 4),
        .default_value = DEF_INITIAL_GAP,
    },
};

// ── Plugin struct ─────────────────────────────────────────────────────────────

struct PitchShifter {
    clap_plugin_t      plugin;
    const clap_host_t* host;
    double semitones   = 0.0;
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

    const auto&    ain   = proc->audio_inputs[0];
    const auto&    aout  = proc->audio_outputs[0];
    const uint32_t nch   = std::min({ain.channel_count, aout.channel_count, 2u});
    const double   ratio = std::pow(2.0, s->semitones / 12.0);

    for (uint32_t c = 0; c < nch; ++c)
        s->ch[c].process(ain.data32[c], aout.data32[c],
                         static_cast<int>(proc->frames_count), ratio, s->initial_gap);

    return CLAP_PROCESS_CONTINUE;
}

static const void* plugin_get_extension(const clap_plugin_t*, const char* id);

// ── Extensions ────────────────────────────────────────────────────────────────

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

static uint32_t latency_get(const clap_plugin_t* p) {
    return static_cast<uint32_t>(self(p)->initial_gap);
}
static const clap_plugin_latency_t s_latency = { latency_get };

static uint32_t par_count(const clap_plugin_t*) { return PARAM_COUNT; }
static bool par_info(const clap_plugin_t*, uint32_t i, clap_param_info_t* o) {
    if (i >= PARAM_COUNT) return false;
    *o = k_params[i]; return true;
}
static bool par_get(const clap_plugin_t* p, clap_id id, double* v) {
    const auto* s = self(p);
    switch (id) {
        case PARAM_SEMITONES:   *v = s->semitones;                      return true;
        case PARAM_INITIAL_GAP: *v = static_cast<double>(s->initial_gap); return true;
    }
    return false;
}
static bool par_to_text(const clap_plugin_t*, clap_id id, double v,
                         char* buf, uint32_t sz) {
    switch (id) {
        case PARAM_SEMITONES:   snprintf(buf, sz, "%.2f st", v);     return true;
        case PARAM_INITIAL_GAP: snprintf(buf, sz, "%d smp", (int)v); return true;
    }
    return false;
}
static bool par_from_text(const clap_plugin_t*, clap_id, const char*, double*) { return false; }
static void par_flush(const clap_plugin_t* p,
                      const clap_input_events_t* in, const clap_output_events_t*) {
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
    .version      = "0.3.0",
    .description  = "OLA + cubic-Hermite resampler pitch shifter",
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
