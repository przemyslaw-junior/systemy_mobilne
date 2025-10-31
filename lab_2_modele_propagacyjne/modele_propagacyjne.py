from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    import pandas as pd
except Exception:
    pd = None


# stałe
TX_POWER_DBM: float = 20.0   # moc nadajnika [dBm]
TX_GAIN_DB: float = 20.0     # zysk anteny Tx [dB]
RX_GAIN_DB: float = 20.0     # zysk anteny Rx [dB]
EXTRA_LOSSES_DB: float = 0.0 # dodatkowe tłumienia (kable itd.) [dB]

# One-Slope / Motley / Multi-Wall
D0_M: float = 1.0                 # odległość referencyjna [m]
GAMMA: float = 4.0                # wykładnik tłumienia (3.5–6 w pomieszczeniach)

# Motley–Keenan – tłumienia przeszkód
BRICK_DB: float = 8.0             # cegła [dB]
CONCRETE_DB: float = 11.0         # beton [dB]

# Multi-Wall – rozróżnienie typów przegród (przykładowe w środku zakresów)
INTERNAL_WALL_DB: float = 7.0     # ściana wewnętrzna [dB]
EXTERNAL_WALL_DB: float = 9.0    # ściana zewnętrzna [dB]
FLOOR_DB: float = 11.0            # strop (między piętrami) [dB]

# ITU-R P.1238 – typowe wartości, jeśli brak tabeli
ITU_N: float = 30.0               # współczynnik tłumienia zależny od środowiska
ITU_LF: float = 15.0              # tłumienie przejścia między kondygnacjami (na piętro) [dB]

# Częstotliwości używane na wykresach (spójne podejście do stałych)
FREQ_24_GHZ_MHZ: float = 2400.0
FREQ_5_GHZ_MHZ: float = 5000.0

# Nazwa pliku wyjściowego (zgodnie ze stylem OUTPUT_PDF z lab_1)
OUTPUT_PDF: str = 'raport_modele_propagacyjne.pdf'


# Funkcje modeli tłumienia L [dB]

def fspl_db(f_mhz: float, d_m: float) -> float:
    d_m = max(d_m, 1e-3)  # unikaj log10(0)
    d_km = d_m / 1000.0
    return 32.44 + 20.0 * math.log10(max(f_mhz, 1e-3)) + 20.0 * math.log10(d_km)

def itu_p1238_db(f_mhz: float, d_m: float, N: float = ITU_N, n_floors: int = 0, Lf: float = ITU_LF) -> float:
    d_m = max(d_m, 1.0)  # standardowo liczony od ~1 m
    return 20.0 * math.log10(max(f_mhz, 1e-3)) + N * math.log10(d_m) + Lf * float(n_floors) - 28.0

def one_slope_db(f_mhz: float, d_m: float, gamma: float = GAMMA, d0_m: float = D0_M) -> float:
    d_m = max(d_m, d0_m)
    L0 = fspl_db(f_mhz, d0_m)
    return L0 + 10.0 * gamma * math.log10(d_m / d0_m)

def motley_keenan_db(
    f_mhz: float,
    d_m: float,
    gamma: float = GAMMA,
    n_brick: int = 0,
    n_concrete: int = 0,
) -> float:
    return one_slope_db(f_mhz, d_m, gamma) + n_brick * BRICK_DB + n_concrete * CONCRETE_DB

def multi_wall_db(
    f_mhz: float,
    d_m: float,
    gamma: float = GAMMA,
    n_internal: int = 0,
    n_external: int = 0,
    n_floors: int = 0,
) -> float:
    return (
        one_slope_db(f_mhz, d_m, gamma)
        + n_internal * INTERNAL_WALL_DB
        + n_external * EXTERNAL_WALL_DB
        + n_floors * FLOOR_DB
    )

# Dane i mapowanie scenariuszy

SCENARIO_ALIASES = {
    'indoors': 'indoors',
    'behind the wall': 'behind the wall',
    'between floors': 'between floors',
    'outdoors': 'outdoors',
}


def normalize_scenario(name: str) -> str:
    key = re.sub(r"\s+", " ", name.strip().lower())
    return SCENARIO_ALIASES.get(key, name.strip())


def scenario_obstacles(scenariusz: str) -> Dict[str, int]:
    s = normalize_scenario(scenariusz)
    if s == 'indoors':
        return dict(n_internal=0, n_external=0, n_floors=0, n_brick=0, n_concrete=0)
    if s == 'behind the wall':
        return dict(n_internal=1, n_external=0, n_floors=0, n_brick=1, n_concrete=0)
    if s == 'between floors':
        return dict(n_internal=0, n_external=0, n_floors=1, n_brick=0, n_concrete=1)  # strop ~ beton
    if s == 'outdoors':
        return dict(n_internal=0, n_external=1, n_floors=0, n_brick=1, n_concrete=0)  # zewnętrzna cegła
    # domyślnie brak przegród
    return dict(n_internal=0, n_external=0, n_floors=0, n_brick=0, n_concrete=0)


# Wczytywanie i przetwarzanie danych

@dataclass
class Pomiar:
    scenariusz: str
    odleglosc_m: float
    czestotliwosc_MHz: float
    RSSI_dBm: float


def parse_pomiary_txt(path: str) -> List[Pomiar]:
    rows: List[Pomiar] = []
    curr_scen: str | None = None
    curr_dist_m: float | None = None

    # Example header: "indoors: 5m distance"
    header_re = re.compile(r"^\s*(.*?)\s*:\s*([0-9]+)\s*m\b", re.IGNORECASE)
    # Example data: "channel 1, frequency - 2412 MHz, ..., RSSI - -50 dBm"
    data_re = re.compile(r"([0-9]{3,5})\s*MHz.*?RSSI\s*-\s*(-?[0-9]+)\s*dBm", re.IGNORECASE)

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            header_m = header_re.search(line)
            if header_m:
                curr_scen = header_m.group(1).strip()
                curr_dist_m = float(header_m.group(2))
                continue

            data_m = data_re.search(line)
            if data_m and curr_scen is not None and curr_dist_m is not None:
                freq_mhz = float(data_m.group(1))
                rssi_dbm = float(data_m.group(2))
                rows.append(Pomiar(curr_scen, curr_dist_m, freq_mhz, rssi_dbm))

    return rows


def to_dataframe(rows: List[Pomiar]):
    dicts = [r.__dict__ for r in rows]
    if pd is None:
        return dicts
    return pd.DataFrame(dicts)

# Obliczenia mocy odebranego i błędów

def received_power_dbm(L_db: float) -> float:
    return TX_POWER_DBM + TX_GAIN_DB + RX_GAIN_DB - L_db - EXTRA_LOSSES_DB


def compute_models_for_row(row: Pomiar) -> Dict[str, float]:
    f = row.czestotliwosc_MHz
    d = row.odleglosc_m
    obs = scenario_obstacles(row.scenariusz)

    L_fspl = fspl_db(f, d)
    L_itu = itu_p1238_db(f, d, N=ITU_N, n_floors=obs.get('n_floors', 0), Lf=ITU_LF)
    L_os = one_slope_db(f, d, gamma=GAMMA)
    L_mk = motley_keenan_db(f, d, gamma=GAMMA, n_brick=obs.get('n_brick', 0), n_concrete=obs.get('n_concrete', 0))
    L_mw = multi_wall_db(
        f,
        d,
        gamma=GAMMA,
        n_internal=obs.get('n_internal', 0),
        n_external=obs.get('n_external', 0),
        n_floors=obs.get('n_floors', 0),
    )

    return {
        'FSPL': received_power_dbm(L_fspl),
        'ITU-R P.1238': received_power_dbm(L_itu),
        'One-Slope': received_power_dbm(L_os),
        'Motley-Keenan': received_power_dbm(L_mk),
        'Multi-Wall': received_power_dbm(L_mw),
    }

def evaluate_errors(rows: List[Pomiar]) -> Tuple[List[Dict], Dict[str, Dict[str, float]]]:
    results = []
    per_model_errors_abs: Dict[str, List[float]] = {}
    per_model_errors_sq: Dict[str, List[float]] = {}

    for r in rows:
        preds = compute_models_for_row(r)
        entry = {
            'scenariusz': normalize_scenario(r.scenariusz),
            'odleglosc_m': r.odleglosc_m,
            'czestotliwosc_MHz': r.czestotliwosc_MHz,
            'RSSI_dBm': r.RSSI_dBm,
        }
        entry.update({f'{k}_Po_dBm': v for k, v in preds.items()})
        for m_name, po in preds.items():
            err = po - r.RSSI_dBm
            per_model_errors_abs.setdefault(m_name, []).append(abs(err))
            per_model_errors_sq.setdefault(m_name, []).append(err * err)
        results.append(entry)

    metrics: Dict[str, Dict[str, float]] = {}
    for m_name in per_model_errors_abs:
        mae = sum(per_model_errors_abs[m_name]) / max(len(per_model_errors_abs[m_name]), 1)
        mse = sum(per_model_errors_sq[m_name]) / max(len(per_model_errors_sq[m_name]), 1)
        rmse = math.sqrt(mse)
        metrics[m_name] = {'MAE': mae, 'RMSE': rmse}

    return results, metrics

# Wykresy i raport PDF

def fig_text_page(title: str, lines: List[str]):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 w calach ~ 210x297 mm
    ax.axis('off')
    y = 0.95
    ax.text(0.5, y, title, ha='center', va='top', fontsize=16, fontweight='bold', transform=ax.transAxes)
    y -= 0.05
    for line in lines:
        ax.text(0.05, y, line, ha='left', va='top', fontsize=10, transform=ax.transAxes)
        y -= 0.03
    return fig

def plot_fspl_comparison(pdf: PdfPages):
    d = [x for x in range(1, 51)]  # 1..50 m
    f_list = [(FREQ_24_GHZ_MHZ, '2.4 GHz'), (FREQ_5_GHZ_MHZ, '5 GHz')]
    plt.figure(figsize=(8, 5))
    for f, label in f_list:
        L = [fspl_db(f, di) for di in d]
        plt.plot(d, L, label=f'FSPL {label}')
    plt.xlabel('Odległość d [m]')
    plt.ylabel('Tłumienie L_FSPL [dB]')
    plt.title('FSPL(d): porównanie 2.4 vs 5 GHz')
    plt.grid(True, alpha=0.3)
    plt.legend()
    pdf.savefig(bbox_inches='tight')
    plt.close()

def plot_models_attenuation(pdf: PdfPages):
    # porównanie tłumienia (L) dla One-Slope, Motley, Multi-Wall – scenariusz „za ścianą”, f=2.4 GHz
    d_vals = [x for x in range(1, 51)]
    f = 2400.0
    obs = scenario_obstacles('behind the wall')

    L_os = [one_slope_db(f, d) for d in d_vals]
    L_mk = [motley_keenan_db(f, d, n_brick=obs['n_brick'], n_concrete=obs['n_concrete']) for d in d_vals]
    L_mw = [multi_wall_db(f, d, n_internal=obs['n_internal'], n_external=obs['n_external'], n_floors=obs['n_floors']) for d in d_vals]

    plt.figure(figsize=(8, 5))
    plt.plot(d_vals, L_os, label='One-Slope')
    plt.plot(d_vals, L_mk, label='Motley-Keenan')
    plt.plot(d_vals, L_mw, label='Multi-Wall')
    plt.xlabel('Odległość d [m]')
    plt.ylabel('Tłumienie L [dB]')
    plt.title('Porównanie tłumienia modeli (za ścianą, 2.4 GHz)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    pdf.savefig(bbox_inches='tight')
    plt.close()


def plot_measured_vs_modeled(pdf: PdfPages, results: List[Dict], metrics: Dict[str, Dict[str, float]]):
    models = ['FSPL', 'ITU-R P.1238', 'One-Slope', 'Motley–Keenan', 'Multi-Wall']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    plt.figure(figsize=(8, 6))
    for m, c in zip(models, colors):
        x = []  # measured
        y = []  # predicted
        for row in results:
            if f'{m}_Po_dBm' in row:
                x.append(row['RSSI_dBm'])
                y.append(row[f'{m}_Po_dBm'])
        if x:
            mae = metrics[m]['MAE']
            plt.scatter(x, y, alpha=0.7, label=f'{m} (MAE={mae:.1f} dB)', color=c)

    # linia idealna y=x
    all_rssi = [row['RSSI_dBm'] for row in results]
    if all_rssi:
        mn, mx = min(all_rssi) - 5, max(all_rssi) + 5
        plt.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, label='Idealnie: y=x')
    plt.xlabel('RSSI (pomiar) [dBm]')
    plt.ylabel('P_o (model) [dBm]')
    plt.title('RSSI (pomiar) vs P_o (model) - wszystkie scenariusze')
    plt.grid(True, alpha=0.3)
    plt.legend()
    pdf.savefig(bbox_inches='tight')
    plt.close()


def build_report_pdf(out_path: str, rows: List[Pomiar], results: List[Dict], metrics: Dict[str, Dict[str, float]]):
    best_model = min(metrics.items(), key=lambda kv: kv[1]['MAE'])[0] if metrics else 'brak'

    with PdfPages(out_path) as pdf:
        lines = [
            'Parametry nadajnika/odbiornika:',
            f'- Pn = {TX_POWER_DBM:.1f} dBm, Gt = {TX_GAIN_DB:.1f} dB, Gr = {RX_GAIN_DB:.1f} dB, A = {EXTRA_LOSSES_DB:.1f} dB',
            '',
            'Modele i stałe:',
            f'- One-Slope: gamma = {GAMMA:.2f}, d0 = {D0_M:.1f} m',
            f'- Motley-Keenan: cegła = {BRICK_DB:.1f} dB, beton = {CONCRETE_DB:.1f} dB',
            f'- Multi-Wall: śc. wewn. = {INTERNAL_WALL_DB:.1f} dB, śc. zewn. = {EXTERNAL_WALL_DB:.1f} dB, strop = {FLOOR_DB:.1f} dB',
            f'- ITU-R P.1238: N = {ITU_N:.1f}, Lf = {ITU_LF:.1f} dB/pietro',
            '',
            'Materiały: ściana wewnętrzna - cegła; zewnętrzna - cegła; strop - beton.',
            'Scenariusze (przyjęte przeszkody):',
            '- wewnątrz: 0 ścian, 0 stropów',
            '- za ścianą: 1 ściana wewnętrzna',
            '- piętro: 1 strop',
            '- na zewnątrz: 1 ściana zewnętrzna',
        ]
        fig = fig_text_page('Modele propagacyjne - Założenia', lines)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        plot_fspl_comparison(pdf)

        plot_models_attenuation(pdf)

        plot_measured_vs_modeled(pdf, results, metrics)

        lines2 = ['Metryki dopasowania (im mniejsze, tym lepiej):']
        for m_name, m in sorted(metrics.items(), key=lambda kv: kv[1]['MAE']):
            lines2.append(f"- {m_name}: MAE = {m['MAE']:.2f} dB, RMSE = {m['RMSE']:.2f} dB")
        lines2.append('')
        lines2.append(f'Najlepiej dopasowany model (MAE): {best_model}')
        lines2.append('')
        lines2.append('Wpływ częstotliwości: 5 GHz ma większe tłumienie (FSPL rośnie z f),')
        lines2.append('co skutkuje zwykle niższym poziomem sygnału niż przy 2.4 GHz.')
        lines2.append('Wpływ odległości: wraz z d rośnie L (logarytmicznie).')
        lines2.append('Wpływ przeszkód: ściany i stropy dodają stałe tłumienia,')
        lines2.append('co dobrze odwzorowują modele Motley-Keenan i Multi-Wall.')
        fig2 = fig_text_page('Podsumowanie i wnioski', lines2)
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'pomiary_lab_2.txt')
    out_pdf = os.path.join(base_dir, OUTPUT_PDF)

    if not os.path.exists(data_path):
        print(f'Nie znaleziono pliku z danymi: {data_path}')
        return

    rows = parse_pomiary_txt(data_path)
    if not rows:
        print('Brak poprawnie wczytanych danych z pliku pomiarowego.')
        return

    results, metrics = evaluate_errors(rows)

    print('Metryki modeli (MAE / RMSE):')
    for name, m in sorted(metrics.items(), key=lambda kv: kv[1]['MAE']):
        print(f"- {name:15s}  MAE = {m['MAE']:.2f} dB   RMSE = {m['RMSE']:.2f} dB")

    if pd is not None:
        df_rows = to_dataframe(rows)
        df_pred = pd.DataFrame(results)
        try:
            exported_csv = os.path.join(base_dir, 'wyniki_modele.csv')
            df_pred.to_csv(exported_csv, index=False, encoding='utf-8')
            print(f'Zapisano wyniki szczegolowe do: {exported_csv}')
        except Exception:
            pass
    else:
        print('(pandas nie jest dostepny - pomijam zapis CSV)')

    build_report_pdf(out_pdf, rows, results, metrics)
    print(f'Zapisano raport PDF: {out_pdf}')


if __name__ == '__main__':
    main()