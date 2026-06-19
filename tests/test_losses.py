def test_losses_module_imports() -> None:
    import pinns.losses.pinn_losses

    assert pinns.losses.pinn_losses is not None
