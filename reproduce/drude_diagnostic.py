"""
Drude診断

目的:
    Figure5 の αFCA が論文より小さい原因を調査する

確認項目:
    Re(ε)_min
    Im(ε)_max
    k_max
    αFCA_max

実行:
    docker run --rm -v "$(pwd):/app" -w /app \
      fslaser-sim python reproduce/drude_diagnostic.py
"""

import sys
import os

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
)

import numpy as np

from modules.euler_fdm.config import (
    EulerFDMConfig,
    GridConfig,
    TimeConfig,
    InitialCondition,
)

from modules.euler_fdm.solver import (
    initialize_state_vectors,
    create_domain_configs,
    compute_safe_dt,
)

from modules.optics.public import compute_laser_field

from modules.material_properties.constants import (
    PHYSICAL,
)

from modules import material_properties

from modules.optics.solver import (
    _compute_drude_epsilon,
    _compute_extinction_coefficient,
)

# --------------------------------------------------
# 設定
# --------------------------------------------------

FLUENCE = 3.06

N_Z = 50
DZ = 5e-7

DT_MAX = 1e-15

T_END = 4.737e-12

RECORD_EVERY = 200


def main():

    config = EulerFDMConfig(
        fluence=FLUENCE,
        grid=GridConfig(
            n_z=N_Z,
            dz=DZ,
        ),
        time=TimeConfig(
            t_end=T_END,
            dt_max=DT_MAX,
            snapshot_interval=200,
        ),
        initial=InitialCondition(),
    )

    optics_cfg, carrier_cfg, ttm_cfg, _ = (
        create_domain_configs(config)
    )

    optics_cfg = optics_cfg.model_copy(
        update={
            "m_eff_drude": PHYSICAL.m_e
        }
    )

    (
        ne,
        Te,
        Tl,
        phase_state,
        latent_acc,
        _,
        _,
    ) = initialize_state_vectors(
        config.grid,
        config.initial,
    )

    t = config.time.t_start
    step = 0

    max_fca = -1.0
    max_data = None

    while t < T_END:

        opt = compute_laser_field(
            ne=ne,
            Tl=Tl,
            Te=Te,
            phase_state=phase_state,
            t=t,
            config=optics_cfg,
        )

        tau_e = material_properties.compute_tau_e(
            ne,
            Te,
            Tl,
            phase_state,
        )

        epsilon = _compute_drude_epsilon(
            ne,
            tau_e,
            optics_cfg,
        )

        k_ext = _compute_extinction_coefficient(
            epsilon
        )

        current_fca = opt.alpha_fca.max()

        if current_fca > max_fca:

            max_fca = current_fca

            max_data = {
                "t_ps":
                    (
                        t
                        - config.time.t_start
                    )
                    * 1e12,

                "ne_max":
                    ne.max(),

                "tau_min_fs":
                    tau_e.min() * 1e15,

                "tau_max_fs":
                    tau_e.max() * 1e15,

                "re_eps_min":
                    epsilon.real.min(),

                "re_eps_max":
                    epsilon.real.max(),

                "im_eps_min":
                    epsilon.imag.min(),

                "im_eps_max":
                    epsilon.imag.max(),

                "k_max":
                    k_ext.max(),

                "alpha_fca_max":
                    current_fca,
            }

        S_max = np.abs(
            opt.source_term
        ).max()

        dt = (
            DT_MAX
            if S_max < 1e6
            else compute_safe_dt(
                Te=Te,
                Tl=Tl,
                ne=ne,
                dz=DZ,
                dt_max=DT_MAX,
                source_term=opt.source_term,
            )
        )

        from modules.carrier.public import (
            advance_carrier_density
        )

        from modules.ttm.public import (
            advance_temperatures
        )

        car = advance_carrier_density(
            ne=ne,
            intensity=opt.intensity,
            Te=Te,
            Tl=Tl,
            phase_state=phase_state,
            dt=dt,
            config=carrier_cfg,
        )

        ttm = advance_temperatures(
            Te=Te,
            Tl=Tl,
            ne=ne,
            dne_dt=car.dne_dt,
            source_term=opt.source_term,
            phase_state=phase_state,
            latent_heat_accumulated=latent_acc,
            dt=dt,
            config=ttm_cfg,
        )

        ne = car.ne
        Te = ttm.Te
        Tl = ttm.Tl

        phase_state = ttm.phase_state
        latent_acc = (
            ttm.latent_heat_accumulated
        )

        t += dt
        step += 1

    print()
    print("=== Drude Diagnostic ===")
    print()

    for k, v in max_data.items():

        if isinstance(v, float):
            print(f"{k:20s}: {v:.6e}")
        else:
            print(f"{k:20s}: {v}")

    print()

    wavelength_cm = (
        optics_cfg.wavelength_nm
        * 1e-7
    )

    expected_alpha = (
        4.0
        * np.pi
        * max_data["k_max"]
        / wavelength_cm
    )

    print(
        "alpha_from_k      : "
        f"{expected_alpha:.6e} cm^-1"
    )

    print(
        "alpha_simulation  : "
        f"{max_data['alpha_fca_max']:.6e} cm^-1"
    )


if __name__ == "__main__":
    main()