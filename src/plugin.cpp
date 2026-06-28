#include <clap/clap.h>
#include <rubberband/RubberBandStretcher.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

using RBS = RubberBand::RubberBandStretcher;

// ── Parameters ────────────────────────────────────────────────────────────────

enum ParamId : clap_id {
    PARAM_PITCH_SEMITONES = 0,
    PARAM_COUNT,
};

static const clap_param_info_t k_params[PARAM_COUNT] = {
    {
        .id            = PARAM_PITCH_SEMITONES,
        .flags         = CLAP_PARAM_IS_AUTOMATABLE | CLAP_PARAM_IS_MODULATABLE,
        .cookie        = nullptr,
        .name          = "Pitch",
        .module        = "",
        .min_value     = -24.0,
        .max_value     =  24.0,
        .default_value =   0.0,
    },
};

// ── Plugin struct ─────────────────────────────────────────────────────────────

struct PitchShifter {
    clap_plugin_t          plugin;
    const clap_host_t*     host;

    double                 sample_rate   = 44100.0;
    uint32_t               max_frames    = 512;
    double                 pitch_semitones = 0.0;   // current value
    double                 pitch_pending   = 0.0;   // set from process thread

    std::unique_ptr<RBS>   stretcher;

    // Per-channel scratch buffers for rubberband
    std::vector<std::vector<float>> input_buf;
    std::vector<const float*>       input_ptrs;
    std::vector<std::vector<float>> output_buf;
    std::vector<float*>             output_ptrs;
};

static PitchShifter* self(const clap_plugin_t* p) {
    return static_cast<PitchShifter*>(p->plugin_data);
}

// ── clap_plugin_t callbacks ───────────────────────────────────────────────────

static bool plugin_init(const clap_plugin_t* p) {
    (void)p;
    return true;
}

static void plugin_destroy(const clap_plugin_t* p) {
    delete self(p);
}

static bool plugin_activate(const clap_plugin_t* p,
                             double sample_rate,
                             uint32_t /*min_frames*/,
                             uint32_t max_frames) {
    auto* s = self(p);
    s->sample_rate = sample_rate;
    s->max_frames  = max_frames;

    constexpr uint32_t channels = 2;

    s->stretcher = std::make_unique<RBS>(
        static_cast<size_t>(sample_rate), channels,
        RBS::OptionProcessRealTime | RBS::OptionPitchHighConsistency);
    s->stretcher->setPitchScale(std::pow(2.0, s->pitch_semitones / 12.0));

    s->input_buf.assign(channels, std::vector<float>(max_frames, 0.f));
    s->input_ptrs.resize(channels);
    s->output_buf.assign(channels, std::vector<float>(max_frames, 0.f));
    s->output_ptrs.resize(channels);
    for (uint32_t c = 0; c < channels; ++c) {
        s->input_ptrs[c]  = s->input_buf[c].data();
        s->output_ptrs[c] = s->output_buf[c].data();
    }

    return true;
}

static void plugin_deactivate(const clap_plugin_t* p) {
    self(p)->stretcher.reset();
}

static bool plugin_start_processing(const clap_plugin_t*) { return true; }
static void plugin_stop_processing(const clap_plugin_t*)  {}

static void plugin_reset(const clap_plugin_t* p) {
    auto* s = self(p);
    if (s->stretcher) s->stretcher->reset();
}

static clap_process_status plugin_process(const clap_plugin_t* p,
                                           const clap_process_t* proc) {
    auto* s = self(p);
    const uint32_t nframes = proc->frames_count;

    // Handle parameter events
    auto* elist = proc->in_events;
    for (uint32_t i = 0; i < elist->size(elist); ++i) {
        auto* hdr = elist->get(elist, i);
        if (hdr->type == CLAP_EVENT_PARAM_VALUE) {
            auto* ev = reinterpret_cast<const clap_event_param_value_t*>(hdr);
            if (ev->param_id == PARAM_PITCH_SEMITONES) {
                s->pitch_semitones = ev->value;
                if (s->stretcher)
                    s->stretcher->setPitchScale(
                        std::pow(2.0, s->pitch_semitones / 12.0));
            }
        }
    }

    if (!s->stretcher || proc->audio_inputs_count < 1 || proc->audio_outputs_count < 1)
        return CLAP_PROCESS_CONTINUE;

    auto& ain  = proc->audio_inputs[0];
    auto& aout = proc->audio_outputs[0];
    const uint32_t channels = std::min({ain.channel_count, aout.channel_count, 2u});

    // Feed input
    for (uint32_t c = 0; c < channels; ++c)
        s->input_ptrs[c] = ain.data32[c];

    s->stretcher->process(s->input_ptrs.data(), nframes, false);

    // Retrieve available output
    int available = s->stretcher->available();
    if (available < 0) available = 0;
    uint32_t to_retrieve = std::min(static_cast<uint32_t>(available), nframes);

    for (uint32_t c = 0; c < channels; ++c)
        s->output_ptrs[c] = aout.data32[c];

    if (to_retrieve > 0) {
        s->stretcher->retrieve(s->output_ptrs.data(), to_retrieve);
    }

    // Zero-pad if rubberband hasn't produced enough output yet (latency build-up)
    for (uint32_t c = 0; c < channels; ++c)
        std::memset(aout.data32[c] + to_retrieve, 0,
                    (nframes - to_retrieve) * sizeof(float));

    return CLAP_PROCESS_CONTINUE;
}

static const void* plugin_get_extension(const clap_plugin_t* p, const char* id);
static void        plugin_on_main_thread(const clap_plugin_t*) {}

// ── Audio ports extension ─────────────────────────────────────────────────────

static uint32_t audio_ports_count(const clap_plugin_t*, bool /*is_input*/) {
    return 1;
}

static bool audio_ports_get(const clap_plugin_t*, uint32_t index,
                             bool /*is_input*/, clap_audio_port_info_t* info) {
    if (index != 0) return false;
    info->id            = 0;
    info->channel_count = 2;
    info->flags         = CLAP_AUDIO_PORT_IS_MAIN;
    info->port_type     = CLAP_PORT_STEREO;
    info->in_place_pair = CLAP_INVALID_ID;
    std::strncpy(info->name, "Main", sizeof(info->name));
    return true;
}

static const clap_plugin_audio_ports_t s_audio_ports = {
    .count = audio_ports_count,
    .get   = audio_ports_get,
};

// ── Params extension ──────────────────────────────────────────────────────────

static uint32_t params_count(const clap_plugin_t*) { return PARAM_COUNT; }

static bool params_get_info(const clap_plugin_t*, uint32_t index,
                             clap_param_info_t* out) {
    if (index >= PARAM_COUNT) return false;
    *out = k_params[index];
    return true;
}

static bool params_get_value(const clap_plugin_t* p, clap_id id, double* val) {
    if (id == PARAM_PITCH_SEMITONES) { *val = self(p)->pitch_semitones; return true; }
    return false;
}

static bool params_value_to_text(const clap_plugin_t*, clap_id id, double val,
                                  char* buf, uint32_t buf_size) {
    if (id == PARAM_PITCH_SEMITONES) {
        snprintf(buf, buf_size, "%.2f st", val);
        return true;
    }
    return false;
}

static bool params_text_to_value(const clap_plugin_t*, clap_id /*id*/,
                                  const char* /*text*/, double* /*out*/) {
    return false;
}

static void params_flush(const clap_plugin_t* p,
                          const clap_input_events_t*  in,
                          const clap_output_events_t* /*out*/) {
    auto* s = self(p);
    for (uint32_t i = 0; i < in->size(in); ++i) {
        auto* hdr = in->get(in, i);
        if (hdr->type == CLAP_EVENT_PARAM_VALUE) {
            auto* ev = reinterpret_cast<const clap_event_param_value_t*>(hdr);
            if (ev->param_id == PARAM_PITCH_SEMITONES)
                s->pitch_semitones = ev->value;
        }
    }
}

static const clap_plugin_params_t s_params = {
    .count          = params_count,
    .get_info       = params_get_info,
    .get_value      = params_get_value,
    .value_to_text  = params_value_to_text,
    .text_to_value  = params_text_to_value,
    .flush          = params_flush,
};

// ── get_extension dispatch ────────────────────────────────────────────────────

static const void* plugin_get_extension(const clap_plugin_t*, const char* id) {
    if (std::strcmp(id, CLAP_EXT_AUDIO_PORTS) == 0) return &s_audio_ports;
    if (std::strcmp(id, CLAP_EXT_PARAMS)      == 0) return &s_params;
    return nullptr;
}

// ── Factory ───────────────────────────────────────────────────────────────────

static const clap_plugin_descriptor_t k_descriptor = {
    .clap_version = CLAP_VERSION_INIT,
    .id           = "com.example.pitch-shifter",
    .name         = "Pitch Shifter",
    .vendor       = "Example",
    .url          = "",
    .manual_url   = "",
    .support_url  = "",
    .version      = "0.1.0",
    .description  = "Pitch shifter using Rubber Band",
    .features     = (const char*[]){
        CLAP_PLUGIN_FEATURE_AUDIO_EFFECT,
        CLAP_PLUGIN_FEATURE_PITCH_SHIFTER,
        nullptr
    },
};

static uint32_t factory_get_plugin_count(const clap_plugin_factory_t*) {
    return 1;
}

static const clap_plugin_descriptor_t* factory_get_plugin_descriptor(
    const clap_plugin_factory_t*, uint32_t index) {
    return index == 0 ? &k_descriptor : nullptr;
}

static const clap_plugin_t* factory_create_plugin(
    const clap_plugin_factory_t*, const clap_host_t* host, const char* plugin_id) {

    if (std::strcmp(plugin_id, k_descriptor.id) != 0) return nullptr;

    auto* s = new PitchShifter();
    s->host = host;
    s->plugin = clap_plugin_t{
        .desc            = &k_descriptor,
        .plugin_data     = s,
        .init            = plugin_init,
        .destroy         = plugin_destroy,
        .activate        = plugin_activate,
        .deactivate      = plugin_deactivate,
        .start_processing = plugin_start_processing,
        .stop_processing  = plugin_stop_processing,
        .reset           = plugin_reset,
        .process         = plugin_process,
        .get_extension   = plugin_get_extension,
        .on_main_thread  = plugin_on_main_thread,
    };
    return &s->plugin;
}

static const clap_plugin_factory_t s_factory = {
    .get_plugin_count      = factory_get_plugin_count,
    .get_plugin_descriptor = factory_get_plugin_descriptor,
    .create_plugin         = factory_create_plugin,
};

// ── Entry point ───────────────────────────────────────────────────────────────

static bool entry_init(const char* /*plugin_path*/) { return true; }
static void entry_deinit() {}

static const void* entry_get_factory(const char* factory_id) {
    if (std::strcmp(factory_id, CLAP_PLUGIN_FACTORY_ID) == 0) return &s_factory;
    return nullptr;
}

CLAP_EXPORT const clap_plugin_entry_t clap_entry = {
    .clap_version = CLAP_VERSION_INIT,
    .init         = entry_init,
    .deinit       = entry_deinit,
    .get_factory  = entry_get_factory,
};
