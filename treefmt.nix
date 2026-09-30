{
  lib,
  pkgs,
  ...
}: let
  inherit (lib.meta) getExe;
in {
  projectRootFile = "flake.nix";
  settings.formatter = {
    ty = {
      command = getExe pkgs.ty;
      options = ["check"];
      includes = ["*.py"];
    };
  };
  programs = {
    # keep-sorted start block=yes
    alejandra.enable = true;
    deadnix.enable = true;
    keep-sorted.enable = true;
    ruff.enable = true;
    statix.enable = true;
    # ty.enable = true;
    # keep-sorted end
  };
}
