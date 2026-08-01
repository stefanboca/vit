{
  inputs.nixpkgs.url = "https://channels.nixos.org/nixos-unstable/nixexprs.tar.xz";

  outputs = {nixpkgs, ...}: let
    inherit (nixpkgs) lib;
    inherit (lib.attrsets) genAttrs;

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
  in {
    devShells = forAllSystems (system: let
      pkgs = nixpkgsFor.${system};
    in {
      default = pkgs.mkShellNoCC {
        buildInputs = [pkgs.cudaPackages_13.cudatoolkit];
        env = {
          NIX_LD_LIBRARY_PATH = "/run/opengl-driver/lib:/run/current-system/sw/share/nix-ld/lib";
        };
      };
    });
  };
}
