{
  inputs = {
    nixpkgs.url = "https://channels.nixos.org/nixos-unstable/nixexprs.tar.xz";

    treefmt-nix.url = "github:numtide/treefmt-nix";
    treefmt-nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    self,
    nixpkgs,
    treefmt-nix,
    ...
  }: let
    inherit (nixpkgs) lib;
    inherit (lib.attrsets) genAttrs mapAttrs' nameValuePair;
    inherit (lib.strings) makeLibraryPath;

    systems = ["x86_64-linux" "aarch64-linux"];
    forAllSystems = genAttrs systems;
    nixpkgsFor = forAllSystems (system:
      import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      });
    treefmtFor = forAllSystems (system: treefmt-nix.lib.evalModule nixpkgsFor.${system} ./treefmt.nix);
  in {
    devShells = forAllSystems (system: let
      pkgs = nixpkgsFor.${system};
    in {
      default = pkgs.mkShellNoCC rec {
        packages = with pkgs; [gcc uv];
        buildInputs = with pkgs; [
          cudaPackages_13.cudatoolkit
          libGL
          libx11
          openssl
          vulkan-loader
          wayland
        ];
        env = {
          LD_LIBRARY_PATH = "/run/opengl-driver/lib:${makeLibraryPath buildInputs}";
          NIX_LD_LIBRARY_PATH = "/run/opengl-driver/lib:/run/current-system/sw/share/nix-ld/lib";
          TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";
        };
      };
    });

    formatter = forAllSystems (system: treefmtFor.${system}.config.build.wrapper);
    checks = forAllSystems (system: let
      devShells = mapAttrs' (n: nameValuePair "devShell-${n}") self.devShells.${system};
      formatting = {formatting = treefmtFor.${system}.config.build.check self;};
    in
      devShells // formatting);
  };
}
