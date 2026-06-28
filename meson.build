project('pitch-shifter', 'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++17',
    'buildtype=debugoptimized',
  ]
)

cpp = meson.get_compiler('cpp')

if not cpp.has_header('clap/clap.h')
  error('CLAP SDK headers not found — enter the Nix dev shell first')
endif

rubberband_dep = dependency('rubberband')

shared_library('pitch-shifter',
  sources: files('src/plugin.cpp'),
  dependencies: [rubberband_dep],
  name_prefix: '',
  name_suffix: 'clap',
  install: true,
  install_dir: get_option('prefix') / 'lib' / 'clap',
)
