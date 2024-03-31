{
  pkgs ? (import <nixpkgs> {
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
      python-pkgs.numba
      python-pkgs.matplotlib
      python-pkgs.cupy
      python-pkgs.scipy
    ]))
    pkgs.julia
    pkgs.cudatoolkit
  ];
}
