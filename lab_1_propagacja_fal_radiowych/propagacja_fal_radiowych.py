import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# stałe i parametry symulacji
SPEED_OF_LIGHT = 3e8      # c = prędkość światła w próżni (m/s)
TX_GAIN = 1.6             # Gt = zysk anteny nadawczej (bez wysokości)
RX_GAIN = 1.6             # Gr = zysk anteny odbiorczej (bez wysokości)
TX_POWER = None         # Pt = moc nadana
RX_POWER = None           # Pr = moc odbierana
FREQ_LOW = 900e6          # f1 = częstotliwość fali (Hz) 900 MHz
FREQ_HIGH = 2.4e9         # f2 = częstotliwość fali (Hz) 2.4 GHz
TX_HEIGHT = 30            # h1 = wysokość anteny nadawczej TX (m)
RX_HEIGHT = 3.0           # h2 = wysokość anteny odbiorczej RX (m)

DIST_SHORT = np.arange(1.0,100.0+0.25, 0.25)   # odległości krótkie (m) 1-100 m (gęsto)
DIST_LONG = np.logspace(0, 4, 2000)            # odległości długie (m) 1-10000 m (rzadko)


def calculate_fspl_ratio(frequency_hz, distance_m):
    '''
    stosunek PR/PT w modelu wolnej przestrzeni (Free-Space Path Loss) bezpośrednio.
    wrór: PR/PT = (Gt *Gr*(lambda/(4*pi*d))^2
    lambda = c/f -> długość fali radiowej
    '''
    wavelength = SPEED_OF_LIGHT / frequency_hz
    return (TX_GAIN * RX_GAIN * (wavelength / (4 * np.pi * distance_m))**2)
    
def calculate_two_ray_ratio(frequency_hz, distance_m):
    '''
    stosunek PR/PT w modelu dwóch promieni (Two-Ray Ground Reflection Model) z odbiciem.
    wrór: PR/PT = (Gt *Gr*(h1*h2/d^2)^2   -> asymptotyczne przybliżenie
    '''
    WAVELENGTH = SPEED_OF_LIGHT / frequency_hz
    direct_path = np.sqrt((TX_HEIGHT - RX_HEIGHT)**2 + distance_m**2)
    reflected_path = np.sqrt((TX_HEIGHT + RX_HEIGHT)**2 + distance_m**2)
    
    phase_direct = -2.0*np.pi*frequency_hz*(direct_path/SPEED_OF_LIGHT)
    phase_reflected = -2.0*np.pi*frequency_hz*(reflected_path/SPEED_OF_LIGHT)
    
    combined_field = (np.exp(1j * phase_direct)/ direct_path) - (np.exp(1j * phase_reflected)/ reflected_path)
    
    return (TX_GAIN * RX_GAIN * (WAVELENGTH / (4.0 * np.pi))**2 * np.abs(combined_field)**2)


def linear_to_db(power_ratio):
    '''
    konwersja liniowa na decybelową
    wrór: L[dB] = 10*log10(Pr / Pt)
    '''
    power_ratio = np.maximum(power_ratio, 1e-300)  # unikanie log(0)
    return 10.0 * np.log10(power_ratio)


CHECK_DISTANCES = np.array([100.0, 10_000.0])
sumary_data = []

for frequency in [FREQ_LOW, FREQ_HIGH]:
    for distance in CHECK_DISTANCES:
        ratio_fspl = calculate_fspl_ratio(frequency, distance)
        ratio_two_ray = calculate_two_ray_ratio(frequency, distance)
        delay_s = distance / SPEED_OF_LIGHT   # opóźnienie w sygnale (s)
        
        sumary_data.append({
            "Frequency_MHz": frequency/1e6,
            "Distance_m": distance,
            "FSPL_dB": linear_to_db(ratio_fspl),
            "TwoRay_dB": linear_to_db(ratio_two_ray),
            "Delay_us": delay_s * 1e6
        })
        
        
def plot_fspl_vs_distance(distatance_array, title_suffix):
    '''
    wykres FSPL dla dwuch częstotliwości
    '''
    plt.plot(distatance_array,linear_to_db(calculate_fspl_ratio(FREQ_LOW, distatance_array)), label=f'FSPL {FREQ_LOW/1e6} MHz')
    plt.plot(distatance_array,linear_to_db(calculate_fspl_ratio(FREQ_HIGH, distatance_array)), label=f'FSPL {FREQ_HIGH/1e6} MHz')
    
    plt.xlabel('Distance (m)')
    plt.ylabel('Path Loss (dB)')
    plt.title(f'Free-Space Path Loss vs Distance {title_suffix}')
    plt.grid(True, linestyle=":")
    plt.legend()

    
def plot_two_ray_vs_distance(distatance_array, title_suffix):
    '''
    wykres Two-Ray dla dwuch częstotliwości
    '''
    plt.plot(distatance_array,linear_to_db(calculate_two_ray_ratio(FREQ_LOW, distatance_array)), label=f'Two-Ray {FREQ_LOW/1e6} MHz')
    plt.plot(distatance_array,linear_to_db(calculate_two_ray_ratio(FREQ_HIGH, distatance_array)), label=f'Two-Ray {FREQ_HIGH/1e6} MHz')
    
    plt.xlabel('Distance (m)')
    plt.ylabel('Path Loss (dB)')
    plt.title(f'Two-Ray Ground Reflection Model vs Distance {title_suffix}')
    plt.grid(True, linestyle=":")
    plt.legend()
    
plots=[]

# (1) FSPL 1-100 m
fig1 = plt.figure(figsize=(8,5))
plot_fspl_vs_distance(DIST_SHORT, '(1-100 m)')
plots.append(fig1)

# (2) FSPL 1-10,000 m (skala logarytmiczna)
fig2 = plt.figure(figsize=(8,5))
plt.semilogx(DIST_LONG, linear_to_db(calculate_fspl_ratio(FREQ_LOW, DIST_LONG)), label=f'FSPL {FREQ_LOW/1e6} MHz')
plt.semilogx(DIST_LONG, linear_to_db(calculate_fspl_ratio(FREQ_HIGH, DIST_LONG)), label=f'FSPL {FREQ_HIGH/1e6} MHz')
plt.xlabel('Distance (m)')
plt.ylabel('Path Loss (dB)')
plt.title('Free-Space Path Loss vs Distance (1-10,000 m)')
plt.grid(True, linestyle=":")
plt.legend()
plots.append(fig2)

# (3) Two-Ray 1-100 m
fig3 = plt.figure(figsize=(8,5))
plot_two_ray_vs_distance(DIST_SHORT, '(1-100 m)')
plots.append(fig3)

# (4) Two-Ray 1-10,000 m (skala logarytmiczna)
fig4 = plt.figure(figsize=(8,5))
plt.semilogx(DIST_LONG, linear_to_db(calculate_two_ray_ratio(FREQ_LOW, DIST_LONG)), label=f'Two-Ray {FREQ_LOW/1e6} MHz')
plt.semilogx(DIST_LONG, linear_to_db(calculate_two_ray_ratio(FREQ_HIGH, DIST_LONG)), label=f'Two-Ray {FREQ_HIGH/1e6} MHz')
plt.xlabel('Distance (m)')
plt.ylabel('Path Loss (dB)')
plt.title('Two-Ray Ground Reflection Model vs Distance (1-10,000 m)')
plt.grid(True, linestyle=":")
plt.legend()
plots.append(fig4)


def generate_conclusions(results):
    '''
    generowanie wniosków na podstawie wyników
    '''
   
    conclusions = []
    conclusions.append("Wnioski z analizy propagacji fal radiowych:\n"
                           '''  Odbicia fal powoduje nakładanie sie sygnałów bezpośrednich i odbitych. Przez to w odbiorniku sygnał może się wzmacniać albo wygaszać, co na wykresach widac jako charakterystyczne 'falowanie' poziomu mocy. To zjawisko nazywa się 'fadlingiem wielodrogowym', czyli częstotliwościowo-selektywnym zanikaniem sygnału.\n'''
                           '''  Wraz ze wzrostem częstotliwości (z 900 MHz do 2.4 GHz) spadek mocy jest większy - tłumienie rośnie proporcjonalnie do kwadratu częstotliwości. Dla tej samej odległości sygnał 2.4 GHz ma więc niższy poziom niż 900 Mhz, a różnice są szczególnie widoczne przy dużych odległościach, np. 10 km.\n'''
                           '''  Odległość ma równiez duży wpływ na spadek mocy. Przy małych odległościach (do ok. 100 m) wyniki z modeli FSPL i dwudrogowego są podobne, ale przy wiekszych dystansach pojawia się wyraźna interferencja (nakładanie się fal). Opóźnienie sygnału rośnie liniowo z odległością (ok.0,33 microsekundy na 100 m).\n'''
                           '''  Falujące wykresy z modelu dwudrogowego wynikają z interferencji fal bezpośrednich i odbitych dlatego pojawiają się minima i maxima mocy, a ich położenie zależy do częstotliwości i odległości. W praktyce takie zjawisko może powodować problemy z jakością odbioru sygnału, szczególnie w środowiskach miejskich z wieloma przeszkodami i odbiciami.\n\n''') 
    for res in results:
        conclusions.append(f"Dla częstotliwości {res['Frequency_MHz']} MHz i odległości {res['Distance_m']} m:\n"
                      f"- Strata w modelu FSPL: {res['FSPL_dB']:.2f} dB\n"
                      f"- Strata w modelu Two-Ray: {res['TwoRay_dB']:.2f} dB\n"
                      f"- Opóźnienie sygnału: {res['Delay_us']:.2f} ns\n")
       
    return "\n".join(conclusions)

conclusions_text = generate_conclusions(sumary_data)

OUTPUT_PDF = 'sprawozdanie_propagacji_fal.pdf'

with PdfPages(OUTPUT_PDF) as pdf:
    
    fig_text = plt.figure(figsize=(8.27, 11.69))  # A4 size in inches
    plt.axis('off')
    plt.text(0.05, 0.95, "Sprawozdanie z Propagacji Fal Radiowych\n\n" + conclusions_text , fontsize=14, va='top', ha='left', wrap=True)
    pdf.savefig(fig_text, bbox_inches='tight')
    plt.close(fig_text)
    
    for fig in plots:
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
print(f'Sprawozdanie zapisane do pliku: {OUTPUT_PDF}')