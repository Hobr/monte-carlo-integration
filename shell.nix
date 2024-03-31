{
  pkgs ? (import <nixpkgs-unstable> {
    config.allowUnfree = true;
    cudaSupport = true;
  }),
  ...
}:
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.isort
      python-pkgs.black
      python-pkgs.flake8
      python-pkgs.pip
      python-pkgs.numpy
      python-pkgs.numbaWithCuda
      python-pkgs.matplotlib
      python-pkgs.cupy
    ]))
    pkgs.julia
    pkgs.cudatoolkit
  ];
}
