{
  description = "Pitch shifter CLAP plugin";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            meson
            ninja
            pkg-config
            clang
            clap
            rubberband
          ];

          shellHook = ''
            echo "pitch-shifter dev shell ready"
          '';
        };
      });
}
