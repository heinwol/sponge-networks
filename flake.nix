{
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

          devEnv = poetry2nixLib.mkPoetryEnv (poetryAttrs // {
            groups = [ "dev" "test" ];
          });

          devEnvPopulated =
            (devEnv.env.overrideAttrs (oldAttrs: rec {
              name = "py";
              buildInputs = with pkgs;
                (oldAttrs.buildInputs or [ ])
                ++ buildInputs-base
                ++ [

                ];
              shellHook = ''
                export MYPYPATH=$PWD/sponge_networks/
              '';
            }));

          sponge-networks = (poetry2nixLib.mkPoetryApplication poetryAttrs).overrideAttrs
            (oldAttrs: rec {
              buildInputs = (oldAttrs.buildInputs or [ ])
                ++ (with pkgs;
                buildInputs-base ++ [

                ]);
            });
        in
        {
          devShells = {
            default = devEnvPopulated;
          };
          packages = {
            devEnv = devEnv;
            default = sponge-networks;
          };
        };
    };
}
