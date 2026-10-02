{
  description = "counterpart — candidate generation + classical consistency scoring for damaged object reconstruction";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      };
      python = pkgs.python312;

      # Runtime libraries that pip wheels (torch, opencv, ...) may link against.
      # Everything else is bundled inside the manylinux wheels.
      runtimeLibs = with pkgs; [
        stdenv.cc.cc.lib # libstdc++, libgcc_s
        zlib
        libGL # libGL.so.1 / libGLX.so.0 (opencv, torch cuDNN deps)
        glib # libglib-2.0, libgthread-2.0
        libglvnd
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "counterpart";

        packages = [
          python
          pkgs.uv
          pkgs.git
          pkgs.nodejs_22
          # Build toolchain for source-built deps (PyPatchMatch, optional extras)
          pkgs.cmake
          pkgs.ninja
          pkgs.pkg-config
        ];

        # NixOS does not expose driver libs in FHS locations; the CUDA userspace
        # driver (libcuda.so) lives here. Needed by torch wheels to reach the GPU.
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs + ":/run/opengl-driver/lib";

        shellHook = ''
          export UV_PYTHON_PREFERENCE=only-system
          export UV_PYTHON_DOWNLOADS=never
          export PYTHONNOUSERSITE=1
          echo "counterpart devShell — python $(python3 --version 2>&1 | cut -d' ' -f2), uv $(uv --version 2>&1 | cut -d' ' -f2)"
        '';
      };
    };
}
