{pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.pandas
      python-pkgs.numpy
      python-pkgs.numba
      python-pkgs.matplotlib
      python-pkgs.isort
      python-pkgs.black
      python-pkgs.flake8
      python-pkgs.pip
      python-pkgs.cupy
      python-pkgs.pycuda
    ]))
    pkgs.julia
  ];
}
