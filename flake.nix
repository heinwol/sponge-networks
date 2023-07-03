rec {
  description = "";

  inputs = {
    nixpkgs.url = "nixpkgs";
  };

  outputs = { self, nixpkgs }:
    let

      system = "x86_64-linux";

      pkgs = nixpkgs.legacyPackages.${system};

      buildInputs = [

      ];

      devShells.default = pkgs.mkShell {
        packages = buildInputs ++ [
          pkgs.poetry
          (pkgs.poetry2nix.mkPoetryEnv {
            projectDir = ./.;
            preferWheels = true; # else it fails

            # for development;
            # TODO: remove runtime dependency
            extraPackages = (p: [ p.python-lsp-server ]);
          })
        ];
        shellHook = ''
          echo "entering dev shell..."
          # eval fish || true
        '';
      };

      package-env = pkgs.poetry2nix.mkPoetryApplication {
        projectDir = ./.;
        preferWheels = true; # else it fails
      };

      sponge-networks = package-env;

    in
    {
      devShells.${system} = devShells;
      packages.${system} = {
        inherit sponge-networks;
        default = sponge-networks;
        # package-env = package-env.dependencyEnv;
      };
    };
}
