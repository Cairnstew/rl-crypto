{
  description = "rl-crypto — unified RL data pipeline over tickerkit + ticker-news-signals";

  # Pinned to the revision already in the local store (matches tickerkit's flake)
  # so `nix develop` / `nix build` are instant on machines that have run it.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/20b1ddd1aa5ace70c9468305030aa4f9ef79671b";

  # uv2nix stack: builds the uv.lock dependency graph from wheels into a hermetic
  # Nix package. ccxt, gnews and vaderSentiment (the upstreams' core deps) are not
  # packaged in nixpkgs, so the runtime deps must come from PyPI — which the
  # lockfile pins exactly. This is the same stack tickerkit uses.
  inputs.pyproject-nix.url = "github:pyproject-nix/pyproject.nix";
  inputs.pyproject-nix.inputs.nixpkgs.follows = "nixpkgs";
  inputs.uv2nix.url = "github:pyproject-nix/uv2nix";
  inputs.uv2nix.inputs.pyproject-nix.follows = "pyproject-nix";
  inputs.uv2nix.inputs.nixpkgs.follows = "nixpkgs";
  inputs.pyproject-build-systems.url = "github:pyproject-nix/build-system-pkgs";
  inputs.pyproject-build-systems.inputs.pyproject-nix.follows = "pyproject-nix";
  inputs.pyproject-build-systems.inputs.uv2nix.follows = "uv2nix";
  inputs.pyproject-build-systems.inputs.nixpkgs.follows = "nixpkgs";

  # The two upstream pipelines, pinned by local git HEAD. uv.lock records them as
  # ../ sibling path deps, so the hermetic build composes a store layout that
  # mirrors this checkout's sibling layout: $out/{rl-crypto,tickerkit,
  # ticker-news-signals}.
  inputs.tickerkit.url = "git+file:../tickerkit?ref=main";
  inputs.ticker-news-signals.url = "git+file:../ticker-news-signals?ref=main";

  outputs =
    { self, nixpkgs, pyproject-nix, uv2nix, pyproject-build-systems, tickerkit, ticker-news-signals }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAll = nixpkgs.lib.genAttrs systems;

      # Python 3.12 to match requires-python and the uv-managed interpreter.
      pythonFor =
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          inherit pkgs;
          python = pkgs.python312;
        };

      # A store layout that puts rl-crypto next to its upstream siblings, so the
      # `../tickerkit` / `../ticker-news-signals` path deps in uv.lock resolve
      # inside the store. rl-crypto's own tree is cleanSource'd (gitignore-aware —
      # never .venv / .uv-cache / *.sqlite3); the upstreams are git+file inputs so
      # they arrive as already-clean git trees.
      composedSource =
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        pkgs.runCommand "rl-crypto-workspace" { } ''
          mkdir -p $out/rl-crypto
          cp -r ${nixpkgs.lib.sources.cleanSource ./.}/. $out/rl-crypto
          cp -r ${tickerkit} $out/tickerkit
          cp -r ${ticker-news-signals} $out/ticker-news-signals
        '';

      # The hermetic package set, built from uv.lock. Shared by `packages` and the
      # helper apps.
      uv2nixFor =
        system:
        let
          inherit (pythonFor system) pkgs python;
          inherit (nixpkgs) lib;
          pythonBase = pkgs.callPackage pyproject-nix.build.packages { inherit python; };
          workspace = uv2nix.lib.workspace.loadWorkspace {
            workspaceRoot = "${composedSource system}/rl-crypto";
          };
          overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
          pythonSet = pythonBase.overrideScope (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              overlay
            ]
          );
          # mkApplication produces a small derivation: the full venv (every
          # lockfile dependency) as context, and only rl-crypto's own bin.
          inherit (pkgs.callPackages pyproject-nix.build.util { }) mkApplication;
          package = mkApplication {
            venv = pythonSet.mkVirtualEnv "rl-crypto-env" workspace.deps.default;
            package = pythonSet.rl-crypto;
          };
        in
        {
          inherit pythonSet package workspace;
        };
    in
    {
      devShells = forAll (
        system:
        let
          inherit (pythonFor system) pkgs python;
        in
        {
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.uv
              # Lint/type tools come from nixpkgs rather than the venv so they are
              # available before `uv sync` has ever run — and, unlike the PyPI
              # wheels, they are runnable binaries on NixOS.
              pkgs.ruff
              pkgs.basedpyright
              pkgs.git
            ];

            # This is the reason the flake exists, same as tickerkit's. `uv`
            # installs PyPI wheels for numpy and pandas, which link against
            # libz.so.1 and libstdc++.so.6 by soname. NixOS has no /lib, so without
            # this everything dies at `import pandas`.
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.zlib
              pkgs.stdenv.cc.cc.lib
            ];

            shellHook = ''
              # Keep uv's cache inside the project.
              export UV_CACHE_DIR="$PWD/.uv-cache"
              export PYTHONDONTWRITEBYTECODE=1

              if [ ! -x .venv/bin/python ]; then
                echo "rl-crypto: creating .venv (first run only)…"
                uv sync --quiet --group dev
              fi

              # The kernelspec Zed looks for; pins .venv/bin/python by absolute path.
              if [ -x .venv/bin/python ] && [ ! -e "$HOME/.local/share/jupyter/kernels/rl-crypto" ]; then
                uv run python tools/install_kernel.py >/dev/null 2>&1 || true
              fi

              # Put the venv first so `python` and `jupyter` are the project's own.
              if [ -d .venv/bin ]; then
                export PATH="$PWD/.venv/bin:$PATH"
              fi

              # The PyPI ruff/basedpyright wheels cannot run on NixOS (dynamically
              # linked); force the nixpkgs-provided binaries in interactive use.
              alias ruff='${pkgs.ruff}/bin/ruff'
              alias basedpyright='${pkgs.basedpyright}/bin/basedpyright'

              echo "rl-crypto dev shell — python $(python -V 2>&1 | cut -d' ' -f2), uv $(uv --version | cut -d' ' -f2)"
              echo "  uv run python -m pytest -m 'not live'   # offline test suite"
              echo "  uv run python tools/build_notebook.py"
              echo "  ruff check src/ tests/ examples/ tools/  (aliased to nixpkgs ruff)"
            '';
          };
        }
      );

      checks = forAll (
        system:
        let
          inherit (pythonFor system) pkgs;
        in
        {
          # `ruff check` only, mirroring tickerkit's sandboxed check. The offline
          # test suite cannot run as a sandboxed Nix build (network-free build, and
          # the venv is what carries pytest) — use `nix develop -c python -m pytest`.
          ruff = pkgs.runCommand "rl-crypto-ruff" { nativeBuildInputs = [ pkgs.ruff ]; } ''
            cd ${self}
            # --no-cache: the source tree is a read-only store path here.
            ruff check --no-cache src/ tests/ examples/ tools/
            touch $out
          '';
        }
      );

      formatter = forAll (system: nixpkgs.legacyPackages.${system}.nixfmt-rfc-style);

      packages = forAll (
        system:
        let
          inherit (uv2nixFor system) package;
        in
        {
          default = package;
          rl-crypto = package;
        }
      );

      apps = forAll (
        system:
        let
          inherit (pythonFor system) pkgs python;
        in
        {
          # The hermetic CLI from uv.lock: `nix run .#default -- unified ETH/USD`.
          default = {
            type = "app";
            program = "${self.packages.${system}.default}/bin/rl-crypto";
          };
          rl-crypto = {
            type = "app";
            program = "${self.packages.${system}.default}/bin/rl-crypto";
          };
          # The production runners: `nix run .#rl-pipeline -- live --tickers BTC/USD`.
          rl-pipeline = {
            type = "app";
            program = "${self.packages.${system}.default}/bin/rl-pipeline";
          };

          # `nix run .#test` executes on the host, so uv may reach the network here
          # even though a Nix *build* may not.
          test = {
            type = "app";
            program = "${pkgs.writeShellApplication {
              name = "rl-crypto-test";
              runtimeInputs = [ python pkgs.uv ];
              text = ''
                uv sync --quiet --group dev
                exec uv run python -m pytest -m "not live" "$@"
              '';
            }}/bin/rl-crypto-test";
          };

          kernel = {
            type = "app";
            program = "${pkgs.writeShellApplication {
              name = "rl-crypto-kernel";
              runtimeInputs = [ python pkgs.uv ];
              text = ''
                uv sync --quiet --group dev
                exec uv run python tools/install_kernel.py
              '';
            }}/bin/rl-crypto-kernel";
          };

          notebook = {
            type = "app";
            program = "${pkgs.writeShellApplication {
              name = "rl-crypto-notebook";
              runtimeInputs = [ python pkgs.uv ];
              text = ''
                uv sync --quiet --group dev
                exec uv run python tools/build_notebook.py
              '';
            }}/bin/rl-crypto-notebook";
          };
        }
      );
    };
}