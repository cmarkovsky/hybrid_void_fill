All results exclude RGI60-14.17287 and its five realizations. Western, Eastern, and combined counts are 1,422, 332, and 1,754 glaciers. Values are rounded to three significant figures. Bold identifies the lowest error metric or bias magnitude within each population, using unrounded values.

**Table 5. Pixel reconstruction errors**

All values are in m a⁻¹.

| Region | Method | MAE | RMSE | NMAD | Bias |
| --- | --- | ---: | ---: | ---: | ---: |
| Western | Constant | 0.265 | 0.372 | 0.279 | -0.0961 |
| Western | Hypsometric | 0.249 | 0.388 | 0.246 | **+0.00470** |
| Western | Hybrid | **0.208** | **0.310** | **0.214** | -0.00982 |
| Eastern | Constant | 0.367 | 0.517 | 0.390 | -0.0835 |
| Eastern | Hypsometric | 0.353 | 0.524 | 0.360 | **+0.00977** |
| Eastern | Hybrid | **0.301** | **0.446** | **0.305** | +0.0466 |
| Combined | Constant | 0.284 | 0.403 | 0.296 | -0.0937 |
| Combined | Hypsometric | 0.269 | 0.417 | 0.264 | +0.00566 |
| Combined | Hybrid | **0.226** | **0.340** | **0.228** | **+0.000864** |

**Table 6. Paired hybrid-minus-hypsometric contrasts**

All values are in m a⁻¹.

| Metric | Western | Eastern |
| --- | ---: | ---: |
| Pixel MAE | -0.0407 [-0.0455, -0.0362] | -0.0519 [-0.0601, -0.0425] |
| Pixel absolute bias | +0.00512 [-0.0143, +0.0175] | +0.0368 [+0.000805, +0.0565] |
| Glacier MAE | -0.0121 [-0.0141, -0.0101] | -0.0136 [-0.0173, -0.00845] |
| Glacier NMAD | -0.00389 [-0.00812, -0.000519] | -0.00909 [-0.0193, +0.00127] |
| Glacier absolute bias | +0.00111 [-0.00588, +0.00654] | +0.0140 [+0.00184, +0.0207] |

**Table 7. Glacier-wide specific reconstruction errors**

All values are in m a⁻¹.

| Region | Method | MAE | RMSE | NMAD | Bias |
| --- | --- | ---: | ---: | ---: | ---: |
| Western | Constant | 0.0721 | 0.101 | 0.0614 | -0.0357 |
| Western | Hypsometric | 0.0601 | 0.0942 | 0.0410 | **+0.00223** |
| Western | Hybrid | **0.0480** | **0.0713** | **0.0371** | -0.00333 |
| Eastern | Constant | 0.0927 | 0.125 | 0.0852 | -0.0303 |
| Eastern | Hypsometric | 0.0802 | 0.116 | 0.0557 | **+0.00423** |
| Eastern | Hybrid | **0.0666** | **0.0961** | **0.0466** | +0.0182 |
| Combined | Constant | 0.0760 | 0.106 | 0.0660 | -0.0347 |
| Combined | Hypsometric | 0.0639 | 0.0987 | 0.0432 | +0.00261 |
| Combined | Hybrid | **0.0515** | **0.0766** | **0.0385** | **+0.000741** |

**Table 8. Integrated volume-rate reconstruction errors**

All values are in m³ a⁻¹.

| Region | Method | MAE | RMSE | Bias |
| --- | --- | ---: | ---: | ---: |
| Western | Constant | 6.81 × 10⁴ | 3.81 × 10⁵ | -3.76 × 10⁴ |
| Western | Hypsometric | 4.55 × 10⁴ | 2.17 × 10⁵ | **+4.13 × 10³** |
| Western | Hybrid | **3.65 × 10⁴** | **2.04 × 10⁵** | +4.92 × 10³ |
| Eastern | Constant | 6.04 × 10⁴ | 1.84 × 10⁵ | -4.01 × 10⁴ |
| Eastern | Hypsometric | 4.11 × 10⁴ | 9.95 × 10⁴ | -7.99 × 10³ |
| Eastern | Hybrid | **3.45 × 10⁴** | **8.98 × 10⁴** | **+5.81 × 10³** |
| Combined | Constant | 6.66 × 10⁴ | 3.52 × 10⁵ | -3.81 × 10⁴ |
| Combined | Hypsometric | 4.47 × 10⁴ | 2.00 × 10⁵ | **+1.83 × 10³** |
| Combined | Hybrid | **3.62 × 10⁴** | **1.87 × 10⁵** | +5.09 × 10³ |

**Table 9. Regional-transfer sensitivity**

Glacier MAE and Bias are in m a⁻¹. Integrated RMSE is in m³ a⁻¹.

| Target | Model | Glacier MAE | Bias | Integrated RMSE |
| --- | --- | ---: | ---: | ---: |
| Western | Constant | 0.0721 | -0.0357 | 3.81 × 10⁵ |
| Western | Hypsometric | 0.0601 | +0.00223 | 2.17 × 10⁵ |
| Western | Hybrid (full data) | **0.0480** | -0.00333 | 2.04 × 10⁵ |
| Western | Hybrid (Western-trained) | 0.0488 | **+0.00121** | **1.55 × 10⁵** |
| Western | Hybrid (Eastern-trained) | 0.0571 | -0.0280 | 2.67 × 10⁵ |
| Eastern | Constant | 0.0927 | -0.0303 | 1.84 × 10⁵ |
| Eastern | Hypsometric | 0.0802 | +0.00423 | 9.95 × 10⁴ |
| Eastern | Hybrid (full data) | 0.0666 | +0.0182 | 8.98 × 10⁴ |
| Eastern | Hybrid (Western-trained) | 0.0737 | +0.0358 | **8.78 × 10⁴** |
| Eastern | Hybrid (Eastern-trained) | **0.0665** | **-0.00343** | 9.02 × 10⁴ |

**Table 10. Coordinate comparison**

MAE and Bias are in m a⁻¹. Vol. RMSE and Vol. bias are in m³ a⁻¹.

| Region | Method / arm | MAE | Bias | Vol. RMSE | Vol. bias |
| --- | --- | ---: | ---: | ---: | ---: |
| Western | Constant | 0.0721 | -0.0357 | 3.81 × 10⁵ | -3.76 × 10⁴ |
| Western | Hypsometric | 0.0601 | +0.00223 | 2.17 × 10⁵ | +4.13 × 10³ |
| Western | Hybrid (coords off) | 0.0493 | -0.0105 | 1.78 × 10⁵ | **-1.66 × 10³** |
| Western | Hybrid (coords on) | **0.0487** | **-0.00166** | **1.78 × 10⁵** | +3.65 × 10³ |
| Eastern | Constant | 0.0927 | -0.0303 | 1.84 × 10⁵ | -4.01 × 10⁴ |
| Eastern | Hypsometric | 0.0802 | +0.00423 | 9.95 × 10⁴ | -7.99 × 10³ |
| Eastern | Hybrid (coords off) | 0.0656 | +0.00721 | 8.44 × 10⁴ | **-8.36 × 10²** |
| Eastern | Hybrid (coords on) | **0.0646** | **+0.000832** | **8.42 × 10⁴** | -4.66 × 10³ |
| Combined | Constant | 0.0760 | -0.0347 | 3.52 × 10⁵ | -3.81 × 10⁴ |
| Combined | Hypsometric | 0.0639 | +0.00261 | 2.00 × 10⁵ | +1.83 × 10³ |
| Combined | Hybrid (coords off) | 0.0524 | -0.00711 | 1.64 × 10⁵ | **-1.50 × 10³** |
| Combined | Hybrid (coords on) | **0.0517** | **-0.00119** | **1.64 × 10⁵** | +2.07 × 10³ |
