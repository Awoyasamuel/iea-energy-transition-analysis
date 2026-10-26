# Raw data

`iea_tes_by_source_world.csv` is an extract of **IEA World Energy Balances**: total energy supply (TES) by source, World, 1990-2024, in terajoules (TJ).

- Source: https://www.iea.org/data-and-statistics/data-product/world-energy-statistics-balances
- Licence: CC BY 4.0 (https://www.iea.org/terms/creative-commons-cc-licenses). Attribution: IEA World Energy Balances.
- File format quirk: two metadata lines and a blank line precede the header; the year column is unnamed. Handled in `src/energy.py::load_tes`.
