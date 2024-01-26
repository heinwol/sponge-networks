rec {
  description = "";

  inputs = {
    nixpkgs.url = "nixpkgs";
    flake-parts.url = "github:hercules-ci/flake-parts";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = inputs.nixpkgs.lib.systems.flakeExposed;

      perSystem = { config, self', inputs', pkgs, system, ... }:
        let
          lib = pkgs.lib;
          poetry2nixLib = (inputs.poetry2nix.lib.mkPoetry2Nix { inherit pkgs; });

          buildInputs-base = with pkgs; [
            graphviz
          ];

          p2n-overrides = poetry2nixLib.defaultPoetryOverrides.extend
            (self: super: {
              clipboard = super.clipboard.overridePythonAttrs (
                old: { buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools ]; }
              );
            });

          poetryAttrs = {
            projectDir = ./.;
            python = pkgs.python311;
            overrides = p2n-overrides;
            preferWheels = true; # I don't want to compile all that
          };

          devEnv = poetry2nixLib.mkPoetryEnv poetryAttrs;

          devEnvPopulated =
            (devEnv.env.overrideAttrs (oldAttrs: rec {
              name = "py";
              buildInputs = with pkgs;
                buildInputs-base ++ [

                ];
              shellHook = ''
                export MYPYPATH=$PWD/sponge_networks/
              '';
            }));

          app = (poetry2nixLib.mkPoetryApplication poetryAttrs).overrideAttrs
            (oldAttrs: rec {
              buildInputs = with pkgs;
                buildInputs-base ++ [

                ];
            });

          sponge-networks = app;

        in
        {
          devShells = {
            default = devEnvPopulated;
          };
          apps = {
            default = sponge-networks;
          };

          packages = {
            devEnv = devEnv;
          };
        };
    };


}
