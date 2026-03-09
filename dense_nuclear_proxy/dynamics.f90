module dense_nuclear_dynamics
  implicit none
contains
  subroutine time_step(psi, n_sites, energy, pairing)
    real(8), intent(inout) :: psi(:)
    integer, intent(in) :: n_sites
    real(8), intent(out) :: energy, pairing
    real(8) :: norm2

    norm2 = sum(psi*psi)
    if (norm2 > 1.0d-15) psi = psi / sqrt(norm2)

    energy = compute_energy(psi)
    pairing = measure_pairing(psi)

    if (n_sites > 0) then
      energy = energy / dble(n_sites)
      pairing = pairing / dble(n_sites)
    end if
  end subroutine time_step

  real(8) function compute_energy(psi)
    real(8), intent(in) :: psi(:)
    compute_energy = sum(psi*psi)
  end function compute_energy

  real(8) function measure_pairing(psi)
    real(8), intent(in) :: psi(:)
    measure_pairing = sum(abs(psi))
  end function measure_pairing
end module dense_nuclear_dynamics
