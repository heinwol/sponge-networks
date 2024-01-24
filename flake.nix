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
      # flake = {
      #   # Put your original flake attributes here.
      # };
      systems = inputs.nixpkgs.lib.systems.flakeExposed;

      perSystem = { config, self', inputs', pkgs, system, ... }:
        let
          # pkgs = (import inputs.nixpkgs {
          #   inherit system;
          #   config.allowUnfree = true;
          # });
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
            }));

          # devShells.default = pkgs.mkShell {
          #   packages = buildInputs ++ [
          #     pkgs.poetry
          #     (pkgs.poetry2nix.mkPoetryEnv {
          #       projectDir = ./.;
          #       python = pkgs.python311;
          #       preferWheels = true; # else it fails
          #       inherit overrides;

          #       # for development;
          #       # TODO: remove runtime dependency
          #       # extraPackages = (p: [ p.python-lsp-server ]);
          #     })
          #   ];
          #   shellHook = ''
          #     echo "entering dev shell..."
          #     # eval fish || true
          #   '';
          # };

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

          # packages.${system} = {
          #   inherit sponge-networks;
          #   default = sponge-networks;
          #   # package-env = package-env.dependencyEnv;
          # };
        };
    };


}
