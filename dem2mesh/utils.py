import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.windows import Window
import plotly.graph_objects as go
from rasterio.enums import Resampling

def jp2_2_tif(file_ingresso, file_uscita): 
    profili_ottimizzati = {
        'driver': 'GTiff',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'compression': 'DEFLATE',
        'predictor': 2
    }

    print("Inizio conversione... Attendi qualche minuto.")
    with rasterio.open(file_ingresso) as src:
        meta = src.meta.copy()
        meta.update(profili_ottimizzati)
        
        with rasterio.open(file_uscita, 'w', **meta) as dst:
            # Legge e scrive a blocchi per non saturare la RAM
            for ji, window in src.block_windows(1):
                data = src.read(1, window=window)
                dst.write(data, 1, window=window)
                
    print("Conversione completata! Usa il nuovo file 'luna_pole_sud_ottimizzato.tif' nei tuoi script.")


def jp2_viewer(file_path):
    with rasterio.open(file_path) as src:
        # Mostra i metadati principali (SRID, trasformazione affine, dimensioni)
        print("Sistema di Riferimento (CRS):", src.crs)
        print("Dimensioni (Pixel):", src.width, "x", src.height)
        print("Numero di bande:", src.count)
        
        dem_data = src.read(1)
        
        nodata_value = src.nodata
        if nodata_value is not None:
            import numpy as np
            dem_data = np.where(dem_data == nodata_value, np.nan, dem_data)


    plt.figure(figsize=(10, 8))
    plt.imshow(dem_data, cmap='terrain')
    plt.colorbar(label='Elevazione (metri)')
    plt.title('Digital Elevation Model da file .jp2')
    plt.show()


def write_and_save_tif(file_ingresso, file_output, lato_pixel):
    with rasterio.open(file_ingresso) as src:
        # 1. Calcolo del centro e della finestra di ritaglio
        centro_x = src.width // 2
        centro_y = src.height // 2
        
        offset_x = centro_x - (lato_pixel // 2)
        offset_y = centro_y - (lato_pixel // 2)
        
        finestra_clipping = Window(offset_x, offset_y, lato_pixel, lato_pixel)
        
        # 2. Lettura dei dati nativi (manteniamo int16 per efficienza e coerenza)
        dem_ritagliato = src.read(1, window=finestra_clipping)
        
        # 3. Aggiornamento dei metadati geografici specifici per il nuovo file
        # Calcola la nuova matrice di trasformazione affine partendo dall'offset
        nuova_trasformazione = rasterio.windows.transform(finestra_clipping, src.transform)
        
        # Copia i metadati del file originale
        nuovi_metadati = src.meta.copy()
        
        # Aggiorna solo i parametri che sono cambiati
        nuovi_metadati.update({
            'height': lato_pixel,
            'width': lato_pixel,
            'transform': nuova_trasformazione
        })

    # 4. Scrittura del nuovo file GeoTIFF sul disco
    with rasterio.open(file_output, 'w', **nuovi_metadati) as dst:
        dst.write(dem_ritagliato, 1)

    print(f"🎉 Nuovo file GeoTIFF salvato con successo: '{file_output}'")
    print(f"Dimensioni: {lato_pixel}x{lato_pixel} pixel | Risoluzione spaziale: 20m/px")


def read_tif(file_path):
    with rasterio.open(file_path) as src:
        print("--- PROPRIETÀ BASE DEL RASTER ---")
        print(f"Larghezza (Colonne):   {src.width} pixel")
        print(f"Altezza (Righe):       {src.height} pixel")
        # print(f"Numero di Bande:       {src.count}")
        # print(f"Tipo di dati (Dtype):  {src.dtypes[0]}") # es. float32, int16
        # print(f"Valore NoData nativo:  {src.nodata}")
        altezza = src.height
        larghezza = src.width
        
        # print("\n--- GEORIFERENZIAZIONE E COORDINATE ---")
        # print(f"Sistema di Riferimento (CRS):\n{src.crs}")
        
        # La risoluzione spaziale (dimensione reale del pixel)
        # Nel tuo caso restituirà (20.0, 20.0) cioè metri per pixel
        print(f"Risoluzione pixel (X, Y): {src.res} metri")
        
        # I confini geografici/metrici estremi della mappa (Bounding Box)
        print(f"Bordi della mappa (Bounds):\n{src.bounds}")
        
        # print("\n--- 📐 MATRICE DI TRASFORMAZIONE AFFINE ---")
        # # La matrice usata dal GIS per convertire i Pixel (riga, colonna) in Metri del mondo reale
        # print(src.transform)
        
        # print("\n--- 📦 DIZIONARIO METADATI COMPLETO (.meta) ---")
        # # Un riassunto rapido in formato dizionario Python, utilissimo per creare nuovi file condivisibili
        # print(src.meta)

        dem_zone = src.read(1)
        quota_minima = np.min(dem_zone)
        quota_massima = np.max(dem_zone)
        quota_media = np.mean(dem_zone)
        deviazione_standard = np.std(dem_zone)
        
        print("\n--- 🏔️ ANALISI STATISTICA ALTEZZE LUNARI ---")
        print(f"Punto più profondo (Crateri): {quota_minima} metri")
        print(f"Punto più elevato (Picchi):     {quota_massima} metri")
        print(f"Dislivello totale nella scena:  {quota_massima - quota_minima} metri")
        print(f"Elevazione media del terreno:   {quota_media:.2f} metri")
        print(f"Variabilità del terreno (Std):  {deviazione_standard:.2f} metri")

        return altezza, larghezza

def visualize_tif(file_path, lato_pixel_vis):
    """
    Legge e visualizza solo una porzione specificata dal centro 
    del file fornito, a piena risoluzione e senza downsampling.
    """
    with rasterio.open(file_path) as src:
        # 1. Calcola il centro geometrico del file corrente (indipendentemente da quanto è grande)
        centro_x = src.width // 2
        centro_y = src.height // 2
        
        # 2. Calcola l'offset per la porzione da visualizzare
        offset_x = centro_x - (lato_pixel_vis // 2)
        offset_y = centro_y - (lato_pixel_vis // 2)
        
        # 3. Applica la finestra di ritaglio visivo e gestisce i confini del file
        finestra_visiva = Window(offset_x, offset_y, lato_pixel_vis, lato_pixel_vis)
        finestra_visiva = finestra_visiva.intersection(Window(0, 0, src.width, src.height))
        
        larghezza_effettiva = finestra_visiva.width
        altezza_effettiva = finestra_visiva.height
        
        if larghezza_effettiva <= 0 or altezza_effettiva <= 0:
            raise ValueError("La dimensione inserita per la visualizzazione non è valida.")
        
        # 4. Lettura dei pixel reali a piena risoluzione
        dem_ritagliato = src.read(1, window=finestra_visiva).astype(float)
        
        # Allinea l'orientamento verticale al GIS standard
        dem_ritagliato = np.flipud(dem_ritagliato)
        
        if src.nodata is not None:
            dem_ritagliato[dem_ritagliato == src.nodata] = np.nan
            
        # 5. Calcola le coordinate chilometriche reali dagli angoli della finestra
        x_min_metri, y_max_metri = src.transform * (offset_x, offset_y)
        x_max_metri, y_min_metri = src.transform * (offset_x + larghezza_effettiva, offset_y + altezza_effettiva)

        # Generazione vettori degli assi per Plotly
        x_km = np.linspace(x_min_metri / 1000.0, x_max_metri / 1000.0, dem_ritagliato.shape[1])
        y_km = np.linspace(y_min_metri / 1000.0, y_max_metri / 1000.0, dem_ritagliato.shape[0])

        print(f"--- 🔍 VISUALIZZAZIONE FINESTRA ---")
        print(f"Risoluzione renderizzata a schermo: {larghezza_effettiva}x{altezza_effettiva} pixel.")

        # 6. Generazione del grafico Plotly Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=dem_ritagliato,
            x=x_km,
            y=y_km,
            colorscale='Earth_r',
            colorbar=dict(title="Elevazione (m)"),
            hovertemplate="X: %{x:.2f} km<br>Y: %{y:.2f} km<br>Quota: %{z:.0f} m<extra></extra>"
        ))

        # Posiziona il marker rosso sul centro esatto della porzione inquadrata
        centro_x_km = (x_km[0] + x_km[-1]) / 2
        centro_y_km = (y_km[0] + y_km[-1]) / 2
        fig.add_trace(go.Scatter(
            x=[centro_x_km], y=[centro_y_km], mode='markers',
            marker=dict(size=12, color="red", line=dict(width=2, color="white")), 
            name="Centro Vista"
        ))

        fig.update_layout(
            title=dict(
                text=f"Mappa 2D | Finestra Rendere: {larghezza_effettiva}x{altezza_effettiva} px<br><span style='font-size: 13px; color: gray;'>Scala reale mostrata: {larghezza_effettiva * 20 / 1000:.2f}x{altezza_effettiva * 20 / 1000:.2f} km</span>",
                x=0.5
            ),
            xaxis=dict(title="Distanza dal Polo Sud - Asse X (km)", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Distanza dal Polo Sud - Asse Y (km)"),
            width=750, height=700,
            margin=dict(l=60, r=40, b=60, t=80)
        )

        fig.show()


def write_and_save_tif(file_ingresso, file_output, lato_pixel, centro_x_km, centro_y_km):
    """
    Esegue il clipping quadrato di un GeoTIFF inserendo il centro in Chilometri (km).
    La conversione in pixel viene gestita internamente basandosi su pixel da 20 metri.
    
    Parametri:
    - file_ingresso: str, percorso file sorgente
    - file_output: str, percorso file destinazione
    - lato_pixel: int, dimensione del lato del quadrato di output (in pixel)
    - centro_x_km: float, coordinata X del centro espressa in km (Est (+) / Ovest (-))
    - centro_y_km: float, coordinata Y del centro espressa in km (Nord (+) / Sud (-))
    """
    risoluzione_metro_pixel = 20.0  # Ogni pixel corrisponde a 20 metri
    
    with rasterio.open(file_ingresso) as src:
        W_max = src.width
        H_max = src.height
        mezzo_lato = lato_pixel // 2
        
        # 1. Trova il centro geometrico del file (Punto 0,0 km della mappa)
        centro_assoluto_x = W_max // 2
        centro_assoluto_y = H_max // 2
        
        # 2. Converti i chilometri passati dall'utente in spostamento pixel (Offset)
        offset_pixel_x = int(round((centro_x_km * 1000.0) / risoluzione_metro_pixel))
        offset_pixel_y = int(round((centro_y_km * 1000.0) / risoluzione_metro_pixel))
        
        # 3. Calcola la posizione esatta del centro in pixel (Indici della matrice)
        # X cresce verso destra (+)
        centro_x = centro_assoluto_x + offset_pixel_x 
        # Y geografico cresce verso l'alto (+), ma gli indici delle righe raster crescono verso il basso.
        # Sottraendo l'offset convertiamo correttamente la coordinata.
        centro_y = centro_assoluto_y - offset_pixel_y 
        
        # --- FAIL-SAFE: Spostamento automatico se la finestra esce dai bordi fisici ---
        offset_x = centro_x - mezzo_lato
        if offset_x < 0:
            offset_x = 0
        elif offset_x + lato_pixel > W_max:
            offset_x = W_max - lato_pixel

        offset_y = centro_y - mezzo_lato
        if offset_y < 0:
            offset_y = 0
        elif offset_y + lato_pixel > H_max:
            offset_y = H_max - lato_pixel
            
        # Controllo strutturale: il ritaglio richiesto non può superare il file originale
        if lato_pixel > W_max or lato_pixel > H_max:
            raise ValueError(f"Errore: Il lato richiesto ({lato_pixel}px) supera la dimensione del file ({W_max}x{H_max}px).")

        # Configurazione della finestra di ritaglio finale protetta
        finestra_clipping = Window(offset_x, offset_y, lato_pixel, lato_pixel)
        
        # Lettura fisica dei dati raster nativi
        dem_ritagliato = src.read(1, window=finestra_clipping)
        
        # Rigenera i metadati spaziali corretti per la nuova sotto-matrice
        nuova_trasformazione = rasterio.windows.transform(finestra_clipping, src.transform)
        nuovi_metadati = src.meta.copy()
        nuovi_metadati.update({
            'height': lato_pixel,
            'width': lato_pixel,
            'transform': nuova_trasformazione
        })

    # Scrittura fisica del file GeoTIFF finale su disco
    with rasterio.open(file_output, 'w', **nuovi_metadati) as dst:
        dst.write(dem_ritagliato, 1)

    print(f"✅ File salvato con successo: '{file_output}'")
    print(f"Coordinate inserite: X = {centro_x_km} km, Y = {centro_y_km} km")
    print(f"Convertite in indici pixel centrali: ({centro_x}, {centro_y})")
    print(f"Finestra applicata ({lato_pixel}x{lato_pixel} px): Colonne [{offset_x}:{offset_x+lato_pixel}], Righe [{offset_y}:{offset_y+lato_pixel}]")

def creates_normals(baricentri, normali, passo_grafico, lunghezza_metri):
    """
    Calcola i segmenti di linea pronti per go.Scatter3d staccati da None,
    ma applica il passo PRIMA di generare le liste per risparmiare memoria e CPU.
    """
    # Applica il campionamento (passo) direttamente sulle matrici dei dati prima del calcolo
    baricentri_ridotti = baricentri[::passo_grafico]
    normali_ridotte = normali[::passo_grafico]
    
    num_tri_filtrati = len(baricentri_ridotti)
    
    # Allocazione matrici vettorizzate
    X_mat = np.zeros((num_tri_filtrati, 3))
    Y_mat = np.zeros((num_tri_filtrati, 3))
    Z_mat = np.zeros((num_tri_filtrati, 3))
    
    p_inizio = baricentri_ridotti
    p_fine = baricentri_ridotti + (normali_ridotte * lunghezza_metri)
    
    X_mat[:, :2] = np.vstack((p_inizio[:, 0], p_fine[:, 0])).T
    Y_mat[:, :2] = np.vstack((p_inizio[:, 1], p_fine[:, 1])).T
    Z_mat[:, :2] = np.vstack((p_inizio[:, 2], p_fine[:, 2])).T
    
    X_mat[:, 2] = np.nan
    Y_mat[:, 2] = np.nan
    Z_mat[:, 2] = np.nan
    
    Xg = [None if np.isnan(val) else val for val in X_mat.flatten()]
    Yg = [None if np.isnan(val) else val for val in Y_mat.flatten()]
    Zg = [None if np.isnan(val) else val for val in Z_mat.flatten()]
    
    return Xg, Yg, Zg

def tif2mesh(file_path, lunghezza_normali_metri, passo, fattore_esagerazione_z = 1):
    """
    Legge un file GeoTIFF e genera esplicitamente l'array dei vertici (X, Y, Z),
    le facce triangolari della mesh, le coordinate per il wireframe (Xe, Ye, Ze),
    le normali unitarie delle facce, i baricentri di ciascun triangolo e le linee delle normali (Xg, Yg, Zg).
    """
    with rasterio.open(file_path) as src:
        Z = src.read(1).astype(float)
        ny, nx = Z.shape
        left, bottom, right, top = src.bounds

    Z = np.flipud(Z)
    Z = Z * fattore_esagerazione_z
        
    # 2. GENERAZIONE DEI VERTICI (X, Y, Z) IN METRI REALI
    x = np.linspace(left, right, nx)
    y = np.linspace(bottom, top, ny)
    X, Y = np.meshgrid(x, y)
    
    vertici_x = X.flatten()
    vertici_y = Y.flatten()
    vertici_z = Z.flatten()
    
    # 3. GENERAZIONE VETTORIZZATA DELLE FACCE (TRIANGOLI)
    indici_griglia = np.arange(ny * nx).reshape(ny, nx)
    
    top_left = indici_griglia[:-1, :-1].flatten()
    top_right = indici_griglia[:-1, 1:].flatten()
    bottom_left = indici_griglia[1:, :-1].flatten()
    bottom_right = indici_griglia[1:, 1:].flatten()
    
    faccia1 = np.vstack((top_left, bottom_left, top_right)).T
    faccia2 = np.vstack((bottom_left, bottom_right, top_right)).T
    facce = np.vstack((faccia1, faccia2))

    p0 = facce[:, 0]
    p1 = facce[:, 1]
    p2 = facce[:, 2]
    
    num_tri = len(facce)
    Xe_mat = np.zeros((num_tri, 5))
    Ye_mat = np.zeros((num_tri, 5))
    Ze_mat = np.zeros((num_tri, 5))
    
    Xe_mat[:, :4] = np.vstack((vertici_x[p0], vertici_x[p1], vertici_x[p2], vertici_x[p0])).T
    Ye_mat[:, :4] = np.vstack((vertici_y[p0], vertici_y[p1], vertici_y[p2], vertici_y[p0])).T
    Ze_mat[:, :4] = np.vstack((vertici_z[p0], vertici_z[p1], vertici_z[p2], vertici_z[p0])).T
    
    Xe_mat[:, 4] = np.nan
    Ye_mat[:, 4] = np.nan
    Ze_mat[:, 4] = np.nan
    
    Xe = [None if np.isnan(val) else val for val in Xe_mat.flatten()]
    Ye = [None if np.isnan(val) else val for val in Ye_mat.flatten()]
    Ze = [None if np.isnan(val) else val for val in Ze_mat.flatten()]
    
    punti_3d = np.vstack((vertici_x, vertici_y, vertici_z)).T
    pt0 = punti_3d[p0]
    pt1 = punti_3d[p1]
    pt2 = punti_3d[p2]
    
    v = pt1 - pt0
    w = pt2 - pt0
    
    normali = np.cross(w, v)
    lunghezze = np.linalg.norm(normali, axis=1, keepdims=True)
    lunghezze = np.where(lunghezze == 0, 1.0, lunghezze)
    normali_unitarie = normali / lunghezze
    
    baricentri = (pt0 + pt1 + pt2) / 3.0
    caratterizza_terreno_da_normali(normali_unitarie, baricentri)

    Xg, Yg, Zg = creates_normals(baricentri, normali_unitarie, passo, lunghezza_metri=lunghezza_normali_metri)
        
    return vertici_x, vertici_y, vertici_z, facce, Xe, Ye, Ze, normali_unitarie, baricentri, Xg, Yg, Zg

def upsampling_geotiff(file_input, file_output, nuova_risoluzione_metri=10.0, metodo_interpolazione='cubic'):
    """
    Esegue l'upsampling di un file GeoTIFF a una risoluzione più fitta scelta dall'utente 
    e salva il nuovo file preservando e aggiornando la georeferenziazione.
    
    Parametri:
    - file_input: str, percorso del file .tif originale (es. il tuo ritaglio da 10km)
    - file_output: str, percorso in cui salvare il nuovo file .tif
    - nuova_risoluzione_metri: float, la nuova dimensione del pixel desiderata (es. 10.0, 5.0)
    - metodo_interpolazione: str, 'cubic' per superfici morbide, 'bilinear' per calcolo veloce
    """
    
    # Mappatura del metodo di interpolazione nativo di Rasterio
    if metodo_interpolazione == 'cubic':
        resampling_algo = Resampling.cubic
    elif metodo_interpolazione == 'cubic_spline':
        resampling_algo = Resampling.cubic_spline
    elif metodo_interpolazione == 'bilinear':
        resampling_algo = Resampling.bilinear
    elif metodo_interpolazione == 'lanczos':
        resampling_algo = Resampling.lanczos
    elif metodo_interpolazione == 'nearest':
        resampling_algo = Resampling.nearest
    elif metodo_interpolazione == 'average':
        resampling_algo = Resampling.average
    elif metodo_interpolazione == 'mode':
        resampling_algo = Resampling.mode
    else:
        raise ValueError(f"Metodo '{metodo_interpolazione}' non supportato o non riconosciuto.")

    with rasterio.open(file_input) as src:
        risoluzione_attuale_x, _ = src.res
        
        # Calcola il fattore di scala basandosi sulla risoluzione scelta
        # Esempio: da 20m a 10m -> fattore = 2.0 (la griglia raddoppia i pixel)
        fattore_scala = risoluzione_attuale_x / nuova_risoluzione_metri
        
        # Calcola le nuove dimensioni della matrice di pixel
        nuova_larghezza = int(src.width * fattore_scala)
        nuova_altezza = int(src.height * fattore_scala)
        
        print(f"--- 🔄 AVVIO RICAMPIONAMENTO ---")
        print(f"Risoluzione originale: {risoluzione_attuale_x} m/px -> Dimensione: {src.width}x{src.height}")
        print(f"Nuova risoluzione:     {nuova_risoluzione_metri} m/px -> Dimensione: {nuova_larghezza}x{nuova_altezza}")
        print(f"Metodo utilizzato:     {metodo_interpolazione}")
        
        # 1. Legge i dati calcolando le quote intermedie con l'algoritmo scelto
        dati_ricampionati = src.read(
            1,
            out_shape=(nuova_altezza, nuova_larghezza),
            resampling=resampling_algo
        )
        
        # 2. Aggiorna la matrice di trasformazione Affine geografica
        # I pixel cambiano dimensione, ma l'angolo in alto a sinistra (Origine) resta fisso nello spazio
        nuova_trasformazione = src.transform * src.transform.scale(
            (src.width / nuova_larghezza),
            (src.height / nuova_altezza)
        )
        
        # 3. Aggiorna il dizionario dei metadati strutturali
        nuovi_metadati = src.meta.copy()
        nuovi_metadati.update({
            "height": nuova_altezza,
            "width": nuova_larghezza,
            "transform": nuova_trasformazione
        })
    
    # 4. Scrive la nuova matrice ad alta densità sul nuovo file .tif
    with rasterio.open(file_output, "w", **nuovi_metadati) as dst:
        dst.write(dati_ricampionati, 1)
        
    print(f"🎉 Salvataggio completato con successo: '{file_output}'\n")

def calcola_normali_facce(vertici_x, vertici_y, vertici_z, facce):
    """
    Calcola i vettori normali unitari (X, Y, Z) per ciascuna faccia triangolare della mesh.
    
    Ritorna:
    - normali: numpy.ndarray di forma (num_facce, 3), contenente [Nx, Ny, Nz] per ogni triangolo.
    """
    # 1. Ricostruiamo la matrice dei vertici [X, Y, Z]
    punti_3d = np.vstack((vertici_x, vertici_y, vertici_z)).T
    
    # 2. Estrarre le coordinate dei tre vertici per ogni triangolo (p0, p1, p2)
    p0 = punti_3d[facce[:, 0]]
    p1 = punti_3d[facce[:, 1]]
    p02 = punti_3d[facce[:, 2]] # chiamo p02 o p2 per coerenza
    
    # 3. Calcolare i vettori dei due lati del triangolo
    v = p1 - p0
    w = p02 - p0
    
    # 4. Prodotto vettoriale (Cross Product) tra i lati per trovare le normali della superficie
    normali = np.cross(v, w)
    
    # 5. Normalizzazione: rendiamo i vettori unitari (lunghezza = 1)
    # Calcoliamo la norma (lunghezza) di ciascun vettore normale
    lunghezze = np.linalg.norm(normali, axis=1, keepdims=True)
    
    # Evitiamo divisioni per zero in caso di triangoli degeneri (piatti/coincidenti)
    lunghezze = np.where(lunghezze == 0, 1.0, lunghezze)
    
    # Dividiamo ciascun vettore per la sua lunghezza
    normali_unitarie = normali / lunghezze
    
    print(f"--- 📐 CALCOLO NORMALI COMPLETATO ---")
    print(f"Matrice delle normali generata con forma: {normali_unitarie.shape}")
    
    return normali_unitarie

def caratterizza_terreno_da_normali(normali_unitarie, baricentri):
    """
    Analizza la pendenza di ciascun triangolo partendo dalle sue normali
    e classifica il terreno in Pianura, Pendio Dolce, Montagna/Versante Ripido.
    Rileva inoltre se la zona si trova sotto o sopra la quota media (Bacino vs Picco).
    """
    # 1. Calcola l'angolo di inclinazione (pendenza) in gradi per ogni faccia
    # Nz è la terza colonna della matrice delle normali (indice 2)
    Nz = normali_unitarie[:, 2]
    
    # Clip di sicurezza per evitare errori di arccos dovuti a precisione floating-point
    Nz_clipped = np.clip(Nz, -1.0, 1.0)
    
    # Angolo rispetto allo zenit in radianti, convertito in gradi
    pendenze_gradi = np.degrees(np.arccos(Nz_clipped))
    
    # 2. Estrai la quota Z dei baricentri per distinguere Bacini (depressioni) da Picchi
    quote_z = baricentri[:, 2]
    quota_media = np.mean(quote_z)
    
    # 3. Classificazione geometrica basata sulle soglie di pendenza
    # Soglie standard GIS: < 5° Pianura, 5°-20° Pendio, > 20° Ripido/Montagna
    pianura_mask = pendenze_gradi < 5.0
    pendio_mask = (pendenze_gradi >= 5.0) & (pendenze_gradi <= 20.0)
    montagna_mask = pendenze_gradi > 20.0
    
    # Classificazione combinata con la quota altimetrica
    bacino_mask = pianura_mask & (quote_z < quota_media)
    altopiano_mask = pianura_mask & (quote_z >= quota_media)
    
    # 4. Calcolo delle statistiche globali della scena
    totale_facce = len(normali_unitarie)
    
    pct_bacino = (np.sum(bacino_mask) / totale_facce) * 100
    pct_altopiano = (np.sum(altopiano_mask) / totale_facce) * 100
    pct_pendio = (np.sum(pendio_mask) / totale_facce) * 100
    pct_montagna = (np.sum(montagna_mask) / totale_facce) * 100
    
    print(f"--- 🌋 ANALISI GEOMORFOLOGICA DEL TERRENO ---")
    print(f"Pendenza Massima Rilevata: {np.max(pendenze_gradi):.1f}°")
    print(f"Pendenza Media Rilevata:  {np.mean(pendenze_gradi):.1f}°")
    print(f"Quota Media di Riferimento: {quota_media:.1f} m\n")
    print(f"Ripartizione della Mesh:")
    print(f" 🟩 Pianura in Depressione (Bacino/Cratere): {pct_bacino:.1f}%")
    print(f" 🟨 Pianura Elevata (Altopiano):             {pct_altopiano:.1f}%")
    print(f" 🟧 Pendii e Raccordi Dolci:                  {pct_pendio:.1f}%")
    print(f" 🟥 Versanti Ripidi (Montagna/Bordi Cratere): {pct_montagna:.1f}%")
    
    # Crea un array di etichette di testo o indici per mappare ogni triangolo
    # classi_facce = np.empty(totale_facce, dtype=object)
    # classi_facce[bacino_mask] = "Bacino"
    # classi_facce[altopiano_mask] = "Altopiano"
    # classi_facce[pendio_mask] = "Pendio"
    # classi_facce[montagna_mask] = "Montagna"
    
    return

def calcola_direzione_raggio(azimuth_deg, elevazione_deg):
    """
    Converte gli angoli di Azimuth ed Elevazione in un vettore 3D unitario.
    - azimuth_deg: 0° (Asse X), 90° (Asse Y), ecc.
    - elevazione_deg: 0° (Orizzontale), -90° (Dritto verso il basso)
    """
    # Conversione in radianti
    azimuth_rad = np.radians(azimuth_deg)
    elevazione_rad = np.radians(elevazione_deg)
    
    # Calcolo delle componenti cartesiane basate sulla trigonometria sferica
    x = np.cos(elevazione_rad) * np.cos(azimuth_rad)
    y = np.cos(elevazione_rad) * np.sin(azimuth_rad)
    z = np.sin(elevazione_rad)
    
    vettore_direzione = np.array([x, y, z])
    
    # Ritorna il vettore normalizzato (unitario)
    return vettore_direzione / np.linalg.norm(vettore_direzione)

def cazzo(dad):
    return