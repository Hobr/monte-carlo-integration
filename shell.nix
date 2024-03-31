{
  pkgs ? (import <nixpkgs-unstable> {
    config.allowUnfree = true;
  }),
  ...
}:
pkgs.mkShell {
  packages = with pkgs; [
    (python3.withPackages (python-pkgs:
      with python-pkgs; [
        isort
        black
        flake8
        pip
        numbaWithCuda
        matplotlib
        cupy
        scipy
      ]))
    julia
  ];
}
