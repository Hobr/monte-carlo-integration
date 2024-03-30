{pkgs ? (import <nixpkgs> {config.allowUnfree = true;}), ...}:
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
    ]))
    pkgs.julia
  ];
}
