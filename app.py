import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib; matplotlib.use('Agg')

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import (chi2_contingency, ttest_ind, f_oneway, kruskal,
                          mannwhitneyu, wilcoxon, pearsonr, spearmanr,
                          shapiro, levene, probplot)
import plotly, plotly.graph_objects as go, plotly.express as px
from plotly.subplots import make_subplots
import json, warnings
warnings.filterwarnings('ignore')

PORT = int(os.environ.get('PORT', 5050))

app = Flask(__name__)

def _to_py(obj):
    """Numpy/pandas tiplerini JSON-serializable Python tiplerine dönüştür."""
    if isinstance(obj, dict):    return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):    return [_to_py(i) for i in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.ndarray):  return [_to_py(x) for x in obj.tolist()]
    return obj

def ok(d): return jsonify(_to_py(d))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Veri & Sabitler ────────────────────────────────────────────────────────────
df_raw = pd.read_csv(os.path.join(BASE_DIR, 'datasetsorunsuz.csv'), encoding='utf-8-sig')
SORU = [f'Soru{i}' for i in range(1, 33)]
df_raw[SORU] = df_raw[SORU].astype(float)

SUBSCALES = {
    'F1: Ayrımcılık & Toksisite':      [f'Soru{i}' for i in range(1, 5)],
    'F2: Bilgi Tehlikeleri':            [f'Soru{i}' for i in range(5, 9)],
    'F3: Yanlış Bilginin Zararı':       [f'Soru{i}' for i in range(9, 14)],
    'F4: Kötüye Kullanımlar':           [f'Soru{i}' for i in range(14, 21)],
    'F5: İnsan-Bilgisayar Etkileşim':   [f'Soru{i}' for i in range(21, 29)],
    'F6: Otomasyon & Çevre Zararları':  [f'Soru{i}' for i in range(29, 33)],
}
SUB_KEYS = list(SUBSCALES.keys())
CAT_COLS = ['Yaş', 'Cinsiyet', 'Fakülte', 'Bölüm', 'Sınıf']
PALETTE  = ['#E53935','#FB8C00','#FDD835','#43A047','#1E88E5','#8E24AA',
            '#00ACC1','#6D4C41','#546E7A','#EC407A']

def get_df():
    d = df_raw.copy()
    for lbl, cols in SUBSCALES.items():
        d[lbl] = d[cols].mean(axis=1)
    d['Etik Farkındalık']         = d[SUB_KEYS[:3]].mean(axis=1)
    d['Sosyal Risk Farkındalığı'] = d[SUB_KEYS[3:]].mean(axis=1)
    d['Genel Farkındalık']        = d[SUB_KEYS].mean(axis=1)
    return d

ALL_NUM = SORU + SUB_KEYS + ['Etik Farkındalık','Sosyal Risk Farkındalığı','Genel Farkındalık']

# ── Yardımcı ───────────────────────────────────────────────────────────────────
def fix_fig(fig):
    """Tüm grafiklerde etiket görünürlüğünü ve marjini düzelt."""
    fig.update_xaxes(
        automargin=True,
        tickangle=-35,
        tickfont=dict(size=11),
        title_font=dict(size=12)
    )
    fig.update_yaxes(
        automargin=True,
        tickfont=dict(size=11),
        title_font=dict(size=12)
    )
    m = fig.layout.margin
    # Üst marjin en az 90, alt marjin en az 120 olsun (dışarı taşan çubuk etiketleri için)
    fig.update_layout(
        margin=dict(
            t = max(int(m.t) if m and m.t else 70, 90),
            b = max(int(m.b) if m and m.b else 0, 120),
            l = max(int(m.l) if m and m.l else 0, 65),
            r = max(int(m.r) if m and m.r else 30, 30),
        ),
        font=dict(family='Segoe UI, system-ui, sans-serif', size=12),
    )
    return fig

def fjson(fig): return json.loads(fix_fig(fig).to_json())

def stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

def sig_badge(p):
    if p < 0.05:
        return f'<span class="badge bg-success">p={p:.4f} — Anlamlı {stars(p)}</span>'
    return f'<span class="badge bg-secondary">p={p:.4f} — Anlamlı değil</span>'

def effect_label(v, kind='v'):
    v = abs(v)
    if kind == 'v':   thresholds = [(0.1,'Önemsiz'),(0.3,'Küçük'),(0.5,'Orta'),(9,'Büyük')]
    elif kind == 'd': thresholds = [(0.2,'Önemsiz'),(0.5,'Küçük'),(0.8,'Orta'),(9,'Büyük')]
    elif kind == 'e': thresholds = [(0.01,'Önemsiz'),(0.06,'Küçük'),(0.14,'Orta'),(9,'Büyük')]
    elif kind == 'r': thresholds = [(0.1,'Önemsiz'),(0.3,'Küçük'),(0.5,'Orta'),(9,'Büyük')]
    for t, lbl in thresholds:
        if v < t: return lbl
    return 'Büyük'

def cronbach(data):
    k = data.shape[1]
    return (k/(k-1)) * (1 - data.var(axis=0,ddof=1).sum() / data.sum(axis=1).var(ddof=1))

def validity_html(is_ok, msg):
    """Testin geçerliliğini gösteren renkli HTML satırı döndürür."""
    if is_ok:
        return f'<br><small class="text-success fw-semibold">✅ <b>Test geçerli:</b> {msg}</small>'
    return f'<br><small class="text-warning fw-semibold">⚠️ <b>Geçerlilik uyarısı:</b> {msg}</small>'

def normality_ok(arr):
    """Shapiro-Wilk ile normallik sınar; True = normal dağılım (p > 0.05)."""
    arr = np.array(arr)
    if len(arr) < 3:
        return True, 1.0          # çok az veri — varsayım test edilemez
    _, p = shapiro(arr[:5000])
    return bool(p > 0.05), float(p)

# ══════════════════════════════════════════════════════════════════════════════
# ROUTE: Ana sayfa
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')

# ── Veri bilgisi ───────────────────────────────────────────────────────────────
@app.route('/api/info')
def api_info():
    df = get_df()
    demo = {}
    for c in CAT_COLS:
        vc = df[c].value_counts()
        demo[c] = {'labels': vc.index.tolist(), 'values': vc.values.tolist()}

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=['Cinsiyet','Yaş','Fakülte','Sınıf','Genel Farkındalık Dağılımı','Alt Boyut Ortalamaları'],
        specs=[[{'type':'pie'},{'type':'bar'},{'type':'bar'}],
               [{'type':'bar'},{'type':'histogram'},{'type':'bar'}]]
    )
    vc = df['Cinsiyet'].value_counts()
    fig.add_trace(go.Pie(labels=vc.index.tolist(), values=vc.values.tolist(),
                         marker_colors=['#2196F3','#E91E63'], hole=0.35,
                         textinfo='label+percent'), row=1, col=1)
    vc = df['Yaş'].value_counts()
    fig.add_trace(go.Bar(x=vc.index.tolist(), y=vc.values.tolist(),
                         marker_color=['#FF9800','#4CAF50'], showlegend=False,
                         text=vc.values.tolist(), textposition='outside'), row=1, col=2)
    vc = df['Fakülte'].value_counts()
    short_fak = [l.replace(' Fakültesi','').replace('İktisadi ve İdari Bilimler','İİBF')
                  .replace('Bilimleri','Bil.') for l in vc.index]
    fig.add_trace(go.Bar(x=short_fak, y=vc.values.tolist(), showlegend=False,
                         marker_color=PALETTE[:len(vc)],
                         text=vc.values.tolist(), textposition='outside'), row=1, col=3)
    sinif = ['1. sınıf','2. sınıf','3. sınıf','4. sınıf']
    vc = df['Sınıf'].value_counts().reindex(sinif, fill_value=0)
    fig.add_trace(go.Bar(x=sinif, y=vc.values.tolist(), showlegend=False,
                         marker_color=['#E53935','#FB8C00','#43A047','#1E88E5'],
                         text=vc.values.tolist(), textposition='outside'), row=2, col=1)
    fig.add_trace(go.Histogram(x=df['Genel Farkındalık'].tolist(), nbinsx=15,
                               marker_color='#42A5F5', showlegend=False), row=2, col=2)
    sub_means = [round(df[c].mean(),3) for c in SUB_KEYS]
    fig.add_trace(go.Bar(x=['F1','F2','F3','F4','F5','F6'], y=sub_means,
                         marker_color=PALETTE[:6], showlegend=False,
                         text=[f'{m:.2f}' for m in sub_means], textposition='outside'), row=2, col=3)
    fig.update_layout(height=600, template='plotly_white',
                      title_text='📊 Veri Genel Bakışı — n=108', showlegend=False,
                      margin=dict(t=80, b=120, l=60, r=30))
    return ok({
        'n': len(df), 'cat_cols': CAT_COLS, 'num_cols': ALL_NUM,
        'subscale_cols': SUB_KEYS, 'demographics': demo,
        'chart': fjson(fig)
    })

# ── Tanımlayıcı İstatistik ─────────────────────────────────────────────────────
@app.route('/api/descriptive', methods=['POST'])
def api_descriptive():
    body = request.json
    var  = body.get('variable')
    grp  = body.get('group_by','none')
    df   = get_df()
    if var not in df.columns:
        return ok({'error': f'"{var}" sütunu bulunamadı'}), 400

    def desc(s):
        return {'n':int(s.count()),'mean':round(float(s.mean()),4),
                'std':round(float(s.std()),4),'sem':round(float(s.sem()),4),
                'median':round(float(s.median()),4),'min':round(float(s.min()),4),
                'max':round(float(s.max()),4),'q1':round(float(s.quantile(.25)),4),
                'q3':round(float(s.quantile(.75)),4),
                'skew':round(float(s.skew()),4),'kurt':round(float(s.kurtosis()),4)}

    if grp and grp != 'none':
        groups  = df[grp].dropna().unique()
        g_stats = {str(g): desc(df[df[grp]==g][var]) for g in groups}
        fig = go.Figure()
        for i,g in enumerate(groups):
            fig.add_trace(go.Box(y=df[df[grp]==g][var].tolist(), name=str(g),
                                 marker_color=PALETTE[i%len(PALETTE)],
                                 boxpoints='all', jitter=0.3, pointpos=-1.8))
        fig.update_layout(template='plotly_white', height=400,
                          title=f'{var} — {grp} Grubuna Göre Dağılım', yaxis_title=var)
        return ok({'type':'grouped','stats':g_stats,'chart':fjson(fig)})
    else:
        s   = desc(df[var])
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Histogram','Kutu Grafiği'])
        fig.add_trace(go.Histogram(x=df[var].tolist(), nbinsx=16,
                                   marker_color='#42A5F5', showlegend=False), row=1, col=1)
        fig.add_trace(go.Box(y=df[var].tolist(), marker_color='#42A5F5',
                             boxpoints='all', jitter=0.3, showlegend=False), row=1, col=2)
        fig.update_layout(template='plotly_white', height=360,
                          title=f'{var} Dağılımı', showlegend=False)
        return ok({'type':'single','stats':s,'chart':fjson(fig)})

# ── Normallik Testi ────────────────────────────────────────────────────────────
@app.route('/api/normality', methods=['POST'])
def api_normality():
    body = request.json
    var  = body.get('variable')
    df   = get_df()
    x    = df[var].dropna()

    sw_stat, sw_p = shapiro(x) if len(x) <= 5000 else (None, None)
    ks_stat, ks_p = stats.kstest(x, 'norm', args=(float(x.mean()), float(x.std())))
    osm, osr      = probplot(x, dist='norm')
    theoretical, ordered = osm[0], osm[1]
    fl_x = [float(min(theoretical)), float(max(theoretical))]
    fl_y = [float(osr[0]*min(theoretical)+osr[1]), float(osr[0]*max(theoretical)+osr[1])]

    fig = make_subplots(rows=1, cols=2, subplot_titles=['Q-Q Grafiği','Histogram + Normal Eğrisi'])
    fig.add_trace(go.Scatter(x=theoretical.tolist(), y=ordered.tolist(), mode='markers',
                             marker=dict(color='#42A5F5', opacity=.7), name='Veri'), row=1, col=1)
    fig.add_trace(go.Scatter(x=fl_x, y=fl_y, mode='lines',
                             line=dict(color='red', width=2), name='Normal'), row=1, col=1)
    hv, be = np.histogram(x, bins=15, density=True)
    xn = np.linspace(float(x.min()), float(x.max()), 100)
    fig.add_trace(go.Bar(x=[(be[i]+be[i+1])/2 for i in range(len(hv))], y=hv.tolist(),
                         marker_color='#42A5F5', opacity=.7, showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=xn.tolist(), y=stats.norm.pdf(xn,float(x.mean()),float(x.std())).tolist(),
                             mode='lines', line=dict(color='red',width=2),
                             showlegend=False), row=1, col=2)
    fig.update_layout(template='plotly_white', height=380, title=f'Normallik — {var}')

    is_normal = (sw_p is not None and sw_p > .05)
    rows_out = []
    if sw_stat: rows_out.append({'test':'Shapiro-Wilk','stat':round(float(sw_stat),4),'p':round(float(sw_p),4),'karar':'Normal ✅' if sw_p>.05 else 'Normal değil ❌'})
    rows_out.append({'test':'Kolmogorov-Smirnov','stat':round(float(ks_stat),4),'p':round(float(ks_p),4),'karar':'Normal ✅' if ks_p>.05 else 'Normal değil ❌'})

    interp = (f'<strong>{var}</strong> için normallik testleri → '
              f'{"✅ Dağılım normale uygundur." if is_normal else "❌ Dağılım normalden sapıyor."} '
              f'Çarpıklık={x.skew():.3f}, Basıklık={x.kurtosis():.3f}.')
    return ok({'rows':rows_out,'is_normal':is_normal,'skew':round(float(x.skew()),4),
                    'kurt':round(float(x.kurtosis()),4),'interpretation':interp,'chart':fjson(fig)})

# ── Ki-Kare ────────────────────────────────────────────────────────────────────
@app.route('/api/chisquare', methods=['POST'])
def api_chisquare():
    body = request.json
    v1, v2 = body.get('var1'), body.get('var2')
    df   = get_df()
    def shorten(lbl):
        return (str(lbl).replace(' Fakültesi','').replace('İktisadi ve İdari Bilimler','İİBF')
                .replace(' Öğretmenliği',' Öğr.').replace(' ve Spor',' & Spor')
                .replace('Bilimleri','Bil.'))

    ct   = pd.crosstab(df[v1], df[v2])
    ct.index   = [shorten(i) for i in ct.index]
    ct.columns = [shorten(c) for c in ct.columns]
    chi2, p, dof, exp = chi2_contingency(ct)
    n    = int(ct.values.sum())
    cv   = float(np.sqrt(chi2 / (n * (min(ct.shape)-1))))
    res  = (ct.values - exp) / np.sqrt(exp)
    low  = int((exp < 5).sum())

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Standartlaşmış Artıklar (|>1.96| = anlamlı)','Satır % Dağılımı'])
    ann_res = [[f'{v:.2f}' for v in row] for row in res.tolist()]
    fig.add_trace(go.Heatmap(z=res.tolist(), x=ct.columns.tolist(),
                             y=[str(i) for i in ct.index.tolist()],
                             colorscale='RdBu_r', zmid=0,
                             text=ann_res, texttemplate='%{text}',
                             colorbar=dict(x=0.46, len=0.9)), row=1, col=1)
    pct = ct.div(ct.sum(axis=1), axis=0)*100
    ann_pct = [[f'{v:.1f}%' for v in row] for row in pct.values.tolist()]
    fig.add_trace(go.Heatmap(z=pct.values.tolist(), x=pct.columns.tolist(),
                             y=[str(i) for i in pct.index.tolist()],
                             colorscale='Blues',
                             text=ann_pct, texttemplate='%{text}'), row=1, col=2)
    fig.update_layout(template='plotly_white', height=400,
                      title=f'Ki-Kare: {v1} × {v2}')

    if low == 0:
        vld = validity_html(True, 'Tüm beklenen frekanslar ≥ 5; ki-kare varsayımı sağlanıyor.')
    else:
        vld = validity_html(False,
            f'{low} hücrede beklenen frekans < 5 (toplam hücrenin %{100*low//ct.size}\'i). '
            f'Fisher\'s Exact testi veya hücre birleştirme düşünülmelidir.')
    interp = (f'<b>{v1} × {v2}</b>: χ²={chi2:.3f}, df={dof}, {sig_badge(p)}. '
              f"Cramér's V={cv:.3f} → <b>{effect_label(cv,'v')}</b> etki.{vld}")
    ct_html = ct.to_html(classes='table table-sm table-bordered text-center', border=0)
    return ok({'chi2':round(chi2,4),'df':dof,'p':round(p,6),'cramers_v':round(cv,4),
                    'effect':effect_label(cv,'v'),'stars':stars(p),'significant':p<.05,
                    'low_exp':low,'ct_html':ct_html,'interpretation':interp,'chart':fjson(fig)})

# ── T-Testi ────────────────────────────────────────────────────────────────────
@app.route('/api/ttest', methods=['POST'])
def api_ttest():
    body  = request.json
    gvar, vvar = body.get('group_var'), body.get('value_var')
    df    = get_df()
    grps  = df[gvar].dropna().unique()
    if len(grps) != 2:
        return ok({'error':f'{gvar} değişkeninde {len(grps)} grup var — t-testi için tam 2 grup gerekli'}), 400

    g1 = df[df[gvar]==grps[0]][vvar].dropna()
    g2 = df[df[gvar]==grps[1]][vvar].dropna()
    lev_s, lev_p = levene(g1, g2)
    eq = lev_p > .05
    t, p = ttest_ind(g1, g2, equal_var=eq)
    pstd = np.sqrt((g1.std()**2+g2.std()**2)/2)
    d    = float((g1.mean()-g2.mean())/pstd) if pstd > 0 else 0.0
    se   = np.sqrt(g1.var(ddof=1)/len(g1) + g2.var(ddof=1)/len(g2))
    df_t = se**4 / ((g1.var(ddof=1)/len(g1))**2/(len(g1)-1) + (g2.var(ddof=1)/len(g2))**2/(len(g2)-1))
    ci   = [round(float((g1.mean()-g2.mean()) + z*se),4) for z in [-1.96,1.96]]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Dağılım (Kutu+Noktalar)','Ortalama ± 95% GA'])
    for i,(g,lbl) in enumerate(zip([g1,g2],grps)):
        fig.add_trace(go.Box(y=g.tolist(), name=str(lbl), marker_color=PALETTE[i],
                             boxpoints='all', jitter=0.3, pointpos=-1.8), row=1, col=1)
    means_ = [float(g1.mean()), float(g2.mean())]
    cis_   = [float(g1.sem()*1.96), float(g2.sem()*1.96)]
    fig.add_trace(go.Bar(x=[str(g) for g in grps], y=means_,
                         error_y=dict(type='data', array=cis_),
                         marker_color=PALETTE[:2], showlegend=False), row=1, col=2)
    fig.update_layout(template='plotly_white', height=400,
                      title=f'T-Testi: {vvar} — {gvar}')

    n1_ok, n1_p = normality_ok(g1)
    n2_ok, n2_p = normality_ok(g2)
    if n1_ok and n2_ok:
        vld = validity_html(True, 'Her iki grup da normal dağılıma uygun (Shapiro-Wilk p > 0.05). T-testi varsayımları sağlanıyor.')
    else:
        vld = validity_html(False,
            f'En az bir grup normal dağılmıyor '
            f'({grps[0]}: p={n1_p:.3f}, {grps[1]}: p={n2_p:.3f}). '
            f'Parametrik olmayan <b>Mann-Whitney U testi</b> tercih edilmelidir.')
    interp = (f'<b>{grps[0]}</b> (n={len(g1)}, Ort={g1.mean():.3f}) vs '
              f'<b>{grps[1]}</b> (n={len(g2)}, Ort={g2.mean():.3f}): '
              f"{'Welch' if not eq else 'Student'} t={t:.3f}, {sig_badge(p)}. "
              f"Cohen's d={d:.3f} → <b>{effect_label(d,'d')}</b> etki. "
              f'%95 GA: [{ci[0]}, {ci[1]}]. '
              f"Levene p={lev_p:.3f} → varyanslar {'eşit ✅' if eq else 'eşit değil ⚠️ (Welch kullanıldı)'}."
              f'{vld}')
    return ok({'t':round(float(t),4),'p':round(float(p),6),'d':round(d,4),
                    'effect':effect_label(d,'d'),'stars':stars(p),'significant':p<.05,
                    'ci':ci,'levene_p':round(float(lev_p),4),'equal_var':bool(eq),
                    'groups':[{'name':str(grps[0]),'n':len(g1),'mean':round(float(g1.mean()),4),'std':round(float(g1.std()),4)},
                               {'name':str(grps[1]),'n':len(g2),'mean':round(float(g2.mean()),4),'std':round(float(g2.std()),4)}],
                    'interpretation':interp,'chart':fjson(fig)})

# ── Mann-Whitney U ─────────────────────────────────────────────────────────────
@app.route('/api/mannwhitney', methods=['POST'])
def api_mannwhitney():
    body  = request.json
    gvar, vvar = body.get('group_var'), body.get('value_var')
    df    = get_df()
    grps  = df[gvar].dropna().unique()
    if len(grps) != 2:
        return ok({'error':f'{gvar} değişkeninde {len(grps)} grup var — 2 grup gerekli'}), 400
    g1 = df[df[gvar]==grps[0]][vvar].dropna()
    g2 = df[df[gvar]==grps[1]][vvar].dropna()
    u, p = mannwhitneyu(g1, g2, alternative='two-sided')
    n1,n2 = len(g1),len(g2)
    r = float(abs((u - n1*n2/2) / np.sqrt(n1*n2*(n1+n2+1)/12)))

    fig = go.Figure()
    for i,(g,lbl) in enumerate(zip([g1,g2],grps)):
        fig.add_trace(go.Box(y=g.tolist(), name=str(lbl), marker_color=PALETTE[i],
                             boxpoints='all', jitter=0.3))
    fig.update_layout(template='plotly_white', height=380,
                      title=f'Mann-Whitney U: {vvar} — {gvar}')
    mw_n1_ok, _ = normality_ok(g1)
    mw_n2_ok, _ = normality_ok(g2)
    if mw_n1_ok and mw_n2_ok:
        vld = validity_html(True, 'Her iki grup da normal dağılıma uygun; parametrik <b>T-testi</b> daha güçlü bir alternatif olabilir. Mann-Whitney yine de geçerlidir.')
    else:
        vld = validity_html(True, 'Dağılım normallik varsayımı gerektirmez. İki bağımsız grup için uygun parametrik olmayan testtir.')
    interp = (f'<b>{grps[0]}</b> (md={g1.median():.3f}) vs <b>{grps[1]}</b> (md={g2.median():.3f}): '
              f'U={u:.1f}, {sig_badge(p)}. r={r:.3f} → <b>{effect_label(r,"r")}</b> etki.{vld}')
    return ok({'U':round(float(u),2),'p':round(float(p),6),'r':round(r,4),
                    'effect':effect_label(r,'r'),'stars':stars(p),'significant':p<.05,
                    'groups':[{'name':str(grps[0]),'n':n1,'median':round(float(g1.median()),4),'mean':round(float(g1.mean()),4)},
                               {'name':str(grps[1]),'n':n2,'median':round(float(g2.median()),4),'mean':round(float(g2.mean()),4)}],
                    'interpretation':interp,'chart':fjson(fig)})

# ── ANOVA ──────────────────────────────────────────────────────────────────────
@app.route('/api/anova', methods=['POST'])
def api_anova():
    body  = request.json
    gvar, vvar = body.get('group_var'), body.get('value_var')
    df    = get_df()
    lbls  = df[gvar].dropna().unique()
    gdata = [df[df[gvar]==g][vvar].dropna().tolist() for g in lbls]
    F, p  = f_oneway(*gdata)
    gm    = float(df[vvar].mean())
    ss_b  = sum(len(g)*(np.mean(g)-gm)**2 for g in gdata)
    ss_t  = sum((x-gm)**2 for g in gdata for x in g)
    eta2  = float(ss_b/ss_t) if ss_t else 0.0
    nc    = len(lbls)*(len(lbls)-1)//2
    posthoc = []
    for i in range(len(lbls)):
        for j in range(i+1,len(lbls)):
            tv, pv = ttest_ind(np.array(gdata[i]), np.array(gdata[j]))
            pb = min(float(pv)*nc, 1.0)
            posthoc.append({'g1':str(lbls[i]),'g2':str(lbls[j]),
                            't':round(float(tv),3),'p_raw':round(float(pv),4),
                            'p_bonf':round(pb,4),'sig':pb<.05})

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Kutu Grafikleri','Ortalama ± 95% GA'])
    for i,(lbl,gd) in enumerate(zip(lbls,gdata)):
        fig.add_trace(go.Box(y=gd, name=str(lbl), marker_color=PALETTE[i%len(PALETTE)],
                             boxpoints='all', jitter=0.3, pointpos=-1.8), row=1, col=1)
    means_ = [np.mean(g) for g in gdata]
    cis_   = [stats.sem(g)*1.96 for g in gdata]
    fig.add_trace(go.Bar(x=[str(l) for l in lbls], y=means_,
                         error_y=dict(type='data', array=cis_),
                         marker_color=PALETTE[:len(lbls)], showlegend=False), row=1, col=2)
    fig.update_layout(template='plotly_white', height=400,
                      title=f'ANOVA: {vvar} — {gvar}')

    n_sig = sum(ph['sig'] for ph in posthoc)
    anova_norm = [normality_ok(g) for g in gdata]
    all_normal = all(ok_ for ok_, _ in anova_norm)
    lev_s2, lev_p2 = levene(*[np.array(g) for g in gdata])
    homogen = lev_p2 > 0.05
    if all_normal and homogen:
        vld = validity_html(True, 'Tüm gruplar normal dağılıma uygun ve varyanslar homojen (Levene p={:.3f}). ANOVA varsayımları sağlanıyor.'.format(lev_p2))
    elif not all_normal:
        non_n = [str(lbls[i]) for i,(ok_,_) in enumerate(anova_norm) if not ok_]
        vld = validity_html(False,
            f'Şu grup(lar) normal dağılmıyor: <b>{", ".join(non_n)}</b>. '
            f'Parametrik olmayan <b>Kruskal-Wallis testi</b> tercih edilmelidir.')
    else:
        vld = validity_html(False,
            f'Varyanslar homojen değil (Levene p={lev_p2:.3f}). '
            f'Welch ANOVA veya <b>Kruskal-Wallis</b> düşünülmelidir.')
    interp = (f'Tek yönlü ANOVA: F({len(lbls)-1},{sum(len(g) for g in gdata)-len(lbls)})={F:.3f}, '
              f'{sig_badge(p)}. η²={eta2:.3f} → <b>{effect_label(eta2,"e")}</b> etki. '
              f'Post-hoc (Bonferroni): {n_sig}/{len(posthoc)} karşılaştırma anlamlı.{vld}')
    gstats = [{'name':str(l),'n':len(g),'mean':round(float(np.mean(g)),4),'std':round(float(np.std(g,ddof=1)),4)} for l,g in zip(lbls,gdata)]
    return ok({'F':round(float(F),4),'p':round(float(p),6),'eta2':round(eta2,4),
                    'effect':effect_label(eta2,'e'),'stars':stars(p),'significant':p<.05,
                    'groups':gstats,'posthoc':posthoc,'interpretation':interp,'chart':fjson(fig)})

# ── Kruskal-Wallis ─────────────────────────────────────────────────────────────
@app.route('/api/kruskal', methods=['POST'])
def api_kruskal():
    body  = request.json
    gvar, vvar = body.get('group_var'), body.get('value_var')
    df    = get_df()
    lbls  = df[gvar].dropna().unique()
    gdata = [df[df[gvar]==g][vvar].dropna().tolist() for g in lbls]
    H, p  = kruskal(*gdata)
    n     = sum(len(g) for g in gdata)
    eps2  = float(H/(n-1))
    nc    = len(lbls)*(len(lbls)-1)//2
    posthoc = []
    for i in range(len(lbls)):
        for j in range(i+1,len(lbls)):
            u, pv = mannwhitneyu(gdata[i], gdata[j], alternative='two-sided')
            pb = min(float(pv)*nc, 1.0)
            posthoc.append({'g1':str(lbls[i]),'g2':str(lbls[j]),
                            'U':round(float(u),1),'p_raw':round(float(pv),4),
                            'p_bonf':round(pb,4),'sig':pb<.05})
    fig = go.Figure()
    for i,(lbl,gd) in enumerate(zip(lbls,gdata)):
        fig.add_trace(go.Box(y=gd, name=str(lbl), marker_color=PALETTE[i%len(PALETTE)],
                             boxpoints='all', jitter=0.3))
    fig.update_layout(template='plotly_white', height=400,
                      title=f'Kruskal-Wallis: {vvar} — {gvar}', yaxis_title=vvar)
    n_sig = sum(ph['sig'] for ph in posthoc)
    kw_norm = [normality_ok(g) for g in gdata]
    kw_all_normal = all(ok_ for ok_, _ in kw_norm)
    if kw_all_normal:
        vld = validity_html(True, 'Tüm gruplar normal dağılıma uygun; parametrik <b>ANOVA</b> daha güçlü olabilir. Kruskal-Wallis yine de geçerlidir.')
    else:
        vld = validity_html(True, 'Normallik varsayımı gerektirmez. Dağılımı normal olmayan çok gruplu karşılaştırmalar için doğru seçimdir.')
    interp = (f'Kruskal-Wallis: H={H:.3f}, {sig_badge(p)}. '
              f'ε²={eps2:.3f}. Post-hoc (MW+Bonferroni): {n_sig}/{len(posthoc)} anlamlı.{vld}')
    return ok({'H':round(float(H),4),'p':round(float(p),6),'epsilon2':round(eps2,4),
                    'stars':stars(p),'significant':p<.05,'posthoc':posthoc,
                    'interpretation':interp,'chart':fjson(fig)})

# ── Korelasyon ─────────────────────────────────────────────────────────────────
@app.route('/api/correlation', methods=['POST'])
def api_correlation():
    body   = request.json
    v1, v2 = body.get('var1'), body.get('var2')
    method = body.get('method','pearson')
    grp    = body.get('group_by','none')
    df     = get_df()
    common = df[[v1,v2]].dropna()
    x, y   = common[v1], common[v2]
    r, p   = (pearsonr(x,y) if method=='pearson' else spearmanr(x,y))
    r, p   = float(r), float(p)
    n      = len(x)
    z      = np.arctanh(r); se = 1/np.sqrt(n-3)
    ci     = [round(float(np.tanh(z-1.96*se)),4), round(float(np.tanh(z+1.96*se)),4)]

    fig = go.Figure()
    if grp and grp != 'none':
        for i,g in enumerate(df[grp].dropna().unique()):
            sub = df[df[grp]==g][[v1,v2]].dropna()
            fig.add_trace(go.Scatter(x=sub[v1].tolist(), y=sub[v2].tolist(),
                                     mode='markers', name=str(g),
                                     marker=dict(color=PALETTE[i], opacity=.75, size=8)))
    else:
        fig.add_trace(go.Scatter(x=x.tolist(), y=y.tolist(), mode='markers',
                                 marker=dict(color='#42A5F5', opacity=.7, size=8), name='Veri'))
    m_, b_ = np.polyfit(x, y, 1)
    xl = np.linspace(float(x.min()),float(x.max()),100)
    fig.add_trace(go.Scatter(x=xl.tolist(), y=(m_*xl+b_).tolist(), mode='lines',
                             line=dict(color='#F44336',width=2.5), name=f'Trend (r={r:.3f})'))
    fig.update_layout(template='plotly_white', height=420,
                      title=f'Korelasyon ({method}): {v1} × {v2}',
                      xaxis_title=v1, yaxis_title=v2)
    rlbl = 'r' if method=='pearson' else 'ρ'
    dir_lbl = ('Güçlü pozitif' if r>.5 else 'Orta pozitif' if r>.3 else
               'Zayıf pozitif' if r>0 else 'Zayıf negatif' if r>-.3 else
               'Orta negatif' if r>-.5 else 'Güçlü negatif')
    if method == 'pearson':
        xn_ok, xn_p = normality_ok(x)
        yn_ok, yn_p = normality_ok(y)
        if xn_ok and yn_ok:
            vld = validity_html(True, 'Her iki değişken de normal dağılıma uygun. Pearson korelasyonu varsayımları sağlanıyor.')
        else:
            vld = validity_html(False,
                f'Değişkenlerden en az biri normal dağılmıyor '
                f'({v1}: p={xn_p:.3f}, {v2}: p={yn_p:.3f}). '
                f'<b>Spearman korelasyonu</b> tercih edilmelidir.')
    else:
        vld = validity_html(True, 'Spearman korelasyonu dağılımdan bağımsızdır; normallik varsayımı gerektirmez.')
    interp = (f'{method.capitalize()}: <b>{rlbl}={r:.4f}</b>, {sig_badge(p)}. '
              f'%95 GA: [{ci[0]}, {ci[1]}]. <b>{dir_lbl}</b> ilişki → {effect_label(r,"r")} etki.{vld}')
    return ok({'r':round(r,4),'p':round(p,6),'method':method,'n':n,'ci':ci,
                    'stars':stars(p),'significant':p<.05,'interpretation':interp,'chart':fjson(fig)})

# ── Korelasyon Matrisi ─────────────────────────────────────────────────────────
@app.route('/api/corrmatrix', methods=['POST'])
def api_corrmatrix():
    body   = request.json
    vars_  = body.get('variables', SUB_KEYS)
    method = body.get('method','pearson')
    df     = get_df()
    corr   = df[vars_].corr(method=method)
    fn     = pearsonr if method=='pearson' else spearmanr
    pmat   = np.ones((len(vars_),len(vars_)))
    for i in range(len(vars_)):
        for j in range(i+1,len(vars_)):
            _, pv = fn(df[vars_[i]].dropna(), df[vars_[j]].dropna())
            pmat[i,j] = pmat[j,i] = pv
    slbls  = [v.split(':')[0] if ':' in v else (v[:12]+'…' if len(v)>12 else v) for v in vars_]
    ann    = [[f"{corr.iloc[i,j]:.2f}{stars(pmat[i,j]) if i!=j else ''}" for j in range(len(vars_))] for i in range(len(vars_))]
    fig = go.Figure(go.Heatmap(z=corr.values.tolist(), x=slbls, y=slbls,
                               colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
                               text=ann, texttemplate='%{text}',
                               colorbar=dict(title='r')))
    fig.update_layout(template='plotly_white', height=500,
                      title=f'Korelasyon Matrisi ({method.capitalize()})')
    return ok({'chart':fjson(fig),'significant':True,
                    'interpretation':f'{method.capitalize()} korelasyon matrisi görüntülendi.'})

# ── Güvenilirlik ───────────────────────────────────────────────────────────────
@app.route('/api/reliability', methods=['POST'])
def api_reliability():
    body  = request.json
    sname = body.get('subscale')
    df    = get_df()
    items = SORU if sname == 'Tümü (32 madde)' else SUBSCALES.get(sname, [])
    if not items:
        return ok({'error':'Geçersiz alt boyut'}), 400
    idata = df[items].dropna()
    alpha = float(cronbach(idata))
    total = idata.sum(axis=1)
    istats = []
    for it in items:
        r_it, _ = pearsonr(idata[it], total)
        rem = idata.drop(columns=[it])
        a_del = float(cronbach(rem)) if len(rem.columns) > 1 else 0.0
        istats.append({'item':it,'mean':round(float(idata[it].mean()),3),
                       'std':round(float(idata[it].std()),3),
                       'r_total':round(float(r_it),3),'alpha_del':round(a_del,3)})
    slbls  = [i['item'].replace('Soru','S') for i in istats]
    colors = ['#43A047' if i['r_total']>=.3 else '#F44336' for i in istats]
    fig = make_subplots(rows=1,cols=2,
                        subplot_titles=['Madde-Toplam Korelasyonları (kesim=0.30)','Silinirse Cronbach α'])
    fig.add_trace(go.Bar(x=slbls, y=[i['r_total'] for i in istats],
                         marker_color=colors, showlegend=False), row=1,col=1)
    fig.add_shape(type='line',x0=-.5,x1=len(items)-.5,y0=.3,y1=.3,
                  line=dict(color='red',dash='dash',width=1.5), row=1,col=1)
    fig.add_trace(go.Bar(x=slbls, y=[i['alpha_del'] for i in istats],
                         marker_color='#42A5F5', showlegend=False), row=1,col=2)
    fig.add_shape(type='line',x0=-.5,x1=len(items)-.5,y0=alpha,y1=alpha,
                  line=dict(color='green',dash='dash',width=1.5), row=1,col=2)
    fig.update_layout(template='plotly_white', height=380,
                      title=f'Güvenilirlik Analizi — {sname} (α={alpha:.3f})')
    albl = ('Mükemmel ≥.90' if alpha>=.9 else 'İyi .80–.90' if alpha>=.8 else
            'Kabul edilebilir .70–.80' if alpha>=.7 else 'Şüpheli .60–.70' if alpha>=.6 else 'Zayıf <.60')
    interp = (f'<b>Cronbach α = {alpha:.3f}</b> → '
              f'<span class="badge {"bg-success" if alpha>=.7 else "bg-warning text-dark" if alpha>=.6 else "bg-danger"}">{albl}</span>. '
              f'{sum(1 for i in istats if i["r_total"]<.3)} maddenin madde-toplam r < 0.30.')
    return ok({'alpha':round(alpha,4),'alpha_label':albl,'n_items':len(items),
                    'n':len(idata),'item_stats':istats,'interpretation':interp,'chart':fjson(fig)})

# ── Alt Boyut Karşılaştırması ──────────────────────────────────────────────────
@app.route('/api/subscale', methods=['POST'])
def api_subscale():
    body = request.json
    grp  = body.get('compare_by','none')
    df   = get_df()
    sub_short = ['F1','F2','F3','F4','F5','F6']
    fig  = make_subplots(rows=1,cols=2,
                         subplot_titles=['Çubuk Grafik (Ortalama ± 95% GA)','Radar Grafiği'],
                         specs=[[{'type':'bar'},{'type':'polar'}]])
    if grp and grp != 'none':
        grp_vals = df[grp].dropna().unique()
        for i,gv in enumerate(grp_vals):
            sub  = df[df[grp]==gv]
            mns  = [float(sub[c].mean()) for c in SUB_KEYS]
            cis_ = [float(sub[c].sem()*1.96) for c in SUB_KEYS]
            fig.add_trace(go.Bar(name=str(gv), x=sub_short, y=mns,
                                 error_y=dict(type='data',array=cis_),
                                 marker_color=PALETTE[i%len(PALETTE)]), row=1,col=1)
            fig.add_trace(go.Scatterpolar(r=mns+[mns[0]], theta=sub_short+[sub_short[0]],
                                          fill='toself', name=str(gv),
                                          line_color=PALETTE[i%len(PALETTE)]), row=1,col=2)
    else:
        mns  = [float(df[c].mean()) for c in SUB_KEYS]
        cis_ = [float(df[c].sem()*1.96) for c in SUB_KEYS]
        fig.add_trace(go.Bar(x=sub_short, y=mns, error_y=dict(type='data',array=cis_),
                             marker_color=PALETTE[:6], showlegend=False,
                             text=[f'{m:.2f}' for m in mns], textposition='outside'), row=1,col=1)
        fig.add_trace(go.Scatterpolar(r=mns+[mns[0]], theta=sub_short+[sub_short[0]],
                                      fill='toself', name='Genel', line_color='#1565C0'), row=1,col=2)
    fig.update_layout(template='plotly_white', height=430, barmode='group',
                      polar=dict(radialaxis=dict(visible=True,range=[0,5])),
                      title='Alt Boyut Karşılaştırması' + (f' — {grp}' if grp!='none' else ''))
    return ok({'chart':fjson(fig),'significant':True,
                    'interpretation':'Alt boyut ortalamaları karşılaştırıldı.'})

# ── Soru Bazlı Toplu Ki-Kare ──────────────────────────────────────────────────
@app.route('/api/chisquare_sorular', methods=['POST'])
def api_chisquare_sorular():
    body   = request.json
    gvar   = body.get('group_var', 'Cinsiyet')
    soru   = body.get('soru', None)   # None → tüm sorular
    df     = get_df()

    def shorten_cat(lbl):
        return (str(lbl).replace(' Fakültesi','').replace('İktisadi ve İdari Bilimler','İİBF')
                .replace(' Öğretmenliği',' Öğr.').replace('Bilimleri','Bil.'))

    target_cols = [soru] if soru else SORU
    results = []
    for col in target_cols:
        # Likert değerlerini (1-5) kategorik olarak kullan
        sub = df[[col, gvar]].dropna().copy()
        sub[col] = sub[col].astype(int).astype(str)   # "1","2","3","4","5"
        ct = pd.crosstab(sub[col], sub[gvar])
        ct.columns = [shorten_cat(c) for c in ct.columns]
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p, dof, exp = chi2_contingency(ct)
        n_tot = int(ct.values.sum())
        cv    = float(np.sqrt(chi2 / (n_tot * (min(ct.shape)-1))))
        low   = int((exp < 5).sum())
        results.append({
            'soru'     : col,
            'chi2'     : round(float(chi2), 3),
            'df'       : int(dof),
            'p'        : round(float(p), 4),
            'cramers_v': round(cv, 3),
            'effect'   : effect_label(cv, 'v'),
            'stars'    : stars(p),
            'significant': bool(p < 0.05),
            'low_exp'  : low,
        })

    if not results:
        return ok({'error': 'Sonuç üretilemedi'}), 400

    # Tek soru → çapraz tablo + heatmap
    if soru:
        r  = results[0]
        sub = df[[soru, gvar]].dropna().copy()
        sub[soru] = sub[soru].astype(int).astype(str)
        ct = pd.crosstab(sub[soru], sub[gvar])
        ct.columns = [shorten_cat(c) for c in ct.columns]
        _, _, _, exp = chi2_contingency(ct)
        res_mat = (ct.values - exp) / np.sqrt(exp)

        fig = make_subplots(rows=1, cols=2,
            subplot_titles=['Standartlaşmış Artıklar (|>1.96|=anlamlı)', 'Satır % Dağılımı'])
        ann_r = [[f'{v:.2f}' for v in row] for row in res_mat.tolist()]
        fig.add_trace(go.Heatmap(z=res_mat.tolist(),
            x=ct.columns.tolist(), y=[f'Puan {i}' for i in ct.index],
            colorscale='RdBu_r', zmid=0,
            text=ann_r, texttemplate='%{text}',
            colorbar=dict(x=0.46, len=0.9)), row=1, col=1)
        pct = ct.div(ct.sum(axis=1), axis=0) * 100
        ann_p = [[f'{v:.1f}%' for v in row] for row in pct.values.tolist()]
        fig.add_trace(go.Heatmap(z=pct.values.tolist(),
            x=pct.columns.tolist(), y=[f'Puan {i}' for i in pct.index],
            colorscale='Blues', text=ann_p, texttemplate='%{text}'), row=1, col=2)
        fig.update_layout(template='plotly_white', height=420,
                          title=f'Ki-Kare: {soru} × {gvar}')

        ct_html = ct.to_html(classes='table table-sm table-bordered text-center', border=0)
        interp = (f'<b>{soru} × {gvar}</b>: χ²={r["chi2"]}, df={r["df"]}, '
                  f'{sig_badge(r["p"])}. Cramér\'s V={r["cramers_v"]} → <b>{r["effect"]}</b> etki.'
                  + (f' ⚠️ {r["low_exp"]} hücrede beklenen <5.' if r['low_exp'] else ''))
        return ok({**r, 'ct_html': ct_html, 'interpretation': interp, 'chart': fjson(fig)})

    # Tüm sorular → özet tablo + Cramér's V bar grafiği
    n_sig = sum(1 for r in results if r['significant'])
    sorular     = [r['soru'] for r in results]
    cramers_vals = [r['cramers_v'] for r in results]
    p_vals       = [r['p'] for r in results]
    bar_colors   = ['#22c55e' if r['significant'] else '#94a3b8' for r in results]

    fig = make_subplots(rows=2, cols=1,
        subplot_titles=[f"Cramér's V (Yeşil = p<0.05 anlamlı, {n_sig}/{len(results)} anlamlı)",
                        'p-değerleri (kesik çizgi = 0.05)'],
        row_heights=[0.55, 0.45])

    fig.add_trace(go.Bar(
        x=[r['soru'].replace('Soru','S') for r in results],
        y=cramers_vals, marker_color=bar_colors,
        text=[f"{r['cramers_v']}{r['stars']}" for r in results],
        textposition='outside', showlegend=False), row=1, col=1)
    fig.add_shape(type='line', x0=-0.5, x1=len(results)-0.5,
                  y0=0.1, y1=0.1, line=dict(color='#FB8C00', dash='dash', width=1.5), row=1, col=1)
    fig.add_shape(type='line', x0=-0.5, x1=len(results)-0.5,
                  y0=0.3, y1=0.3, line=dict(color='#E53935', dash='dash', width=1.5), row=1, col=1)

    fig.add_trace(go.Bar(
        x=[r['soru'].replace('Soru','S') for r in results],
        y=p_vals, marker_color=bar_colors, showlegend=False,
        text=[f"{r['p']}" for r in results], textposition='outside'), row=2, col=1)
    fig.add_shape(type='line', x0=-0.5, x1=len(results)-0.5,
                  y0=0.05, y1=0.05, line=dict(color='#E53935', dash='dash', width=1.5), row=2, col=1)

    fig.update_layout(template='plotly_white', height=620,
                      title=f'32 Soru × {gvar} — Ki-Kare Toplu Analizi')

    interp = (f'<b>{gvar}</b> ile 32 madde arasında ki-kare testi uygulandı. '
              f'<b>{n_sig}/{len(results)}</b> maddede p<0.05 anlamlı fark bulundu. '
              f'En güçlü ilişki: <b>{max(results, key=lambda r: r["cramers_v"])["soru"]}</b> '
              f'(V={max(results, key=lambda r: r["cramers_v"])["cramers_v"]}).')
    return ok({'results': results, 'n_sig': n_sig, 'n_total': len(results),
               'interpretation': interp, 'chart': fjson(fig)})


# ── Wilcoxon İşaretli Sıra Testi ──────────────────────────────────────────────
@app.route('/api/wilcoxon', methods=['POST'])
def api_wilcoxon():
    body  = request.json
    v1    = body.get('var1')     # ilk değişken (ya da tek değişken, y=0 varsayımı)
    v2    = body.get('var2', None)  # ikinci değişken (eşleştirilmiş karşılaştırma)
    mu    = float(body.get('mu', 0))  # tek değişken için test değeri
    df    = get_df()

    if v2 and v2 != 'none':
        # Eşleştirilmiş karşılaştırma: v1 − v2
        common = df[[v1, v2]].dropna()
        x1, x2 = common[v1].values, common[v2].values
        diff = x1 - x2
        label = f'{v1} − {v2}'
    else:
        # Tek örneklem: v1 − mu
        x1 = df[v1].dropna().values
        x2 = None
        diff = x1 - mu
        label = f'{v1} (H₀: medyan = {mu})'

    # Sıfır farkları çıkar
    diff_nz = diff[diff != 0]
    n = len(diff_nz)
    if n < 5:
        return ok({'error': 'Yeterli sıfır-dışı fark yok (n<5). Değişkenleri kontrol edin.'}), 400

    w_stat, p = wilcoxon(diff_nz)
    # Etki büyüklüğü r = Z / √n
    from scipy.stats import norm as _norm
    z_val  = float(_norm.ppf(p / 2))   # yaklaşık Z
    r_eff  = float(abs(z_val) / np.sqrt(n))

    # Pozitif / negatif / sıfır sıra sayıları
    pos  = int((diff > 0).sum())
    neg  = int((diff < 0).sum())
    zero = int((diff == 0).sum())

    # Görselleştirme: fark dağılımı + histogram
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=['Fark Dağılımı (v1 − v2)', 'Fark Histogramı'])
    fig.add_trace(go.Box(y=diff.tolist(), name=label,
                         marker_color='#42A5F5', boxpoints='all',
                         jitter=0.3, pointpos=-1.8), row=1, col=1)
    fig.add_shape(type='line', x0=-0.5, x1=0.5, y0=mu, y1=mu,
                  line=dict(color='red', dash='dash', width=2), row=1, col=1)
    fig.add_trace(go.Histogram(x=diff.tolist(), nbinsx=15,
                               marker_color='#42A5F5', showlegend=False), row=1, col=2)
    fig.add_shape(type='line', x0=0, x1=0, y0=0,
                  y1=max(np.histogram(diff, bins=15)[0]),
                  line=dict(color='red', dash='dash', width=2), row=1, col=2)
    fig.update_layout(template='plotly_white', height=420,
                      title=f'Wilcoxon İşaretli Sıra: {label}')

    sig = p < 0.05
    sig_msg = "Medyan fark 0'dan anlamlı biçimde farklı." if sig else "Medyan fark 0'dan anlamlı biçimde farklı değil."
    if n >= 10:
        vld = validity_html(True, f'Yeterli gözlem sayısı (n={n}, sıfır hariç). Parametrik olmayan eşleştirilmiş karşılaştırma için uygundur.')
    else:
        vld = validity_html(False, f'Sıfır-dışı fark sayısı az (n={n}). Sonuçlar ihtiyatla yorumlanmalıdır; en az 10 gözlem önerilir.')
    interp = (
        f'<b>Wilcoxon İşaretli Sıra Testi</b> — {label}: '
        f'W={w_stat:.1f}, {sig_badge(p)}. '
        f'r={r_eff:.3f} → <b>{effect_label(r_eff, "r")}</b> etki. '
        f'Pozitif fark: {pos}, Negatif fark: {neg}, Sıfır: {zero}. {sig_msg}{vld}'
    )
    return ok({
        'W': round(float(w_stat), 3), 'p': round(float(p), 6),
        'r': round(r_eff, 4), 'effect': effect_label(r_eff, 'r'),
        'stars': stars(p), 'significant': sig, 'n': n,
        'pos': pos, 'neg': neg, 'zero': zero,
        'label': label, 'interpretation': interp, 'chart': fjson(fig)
    })


if __name__ == '__main__':
    debug = os.environ.get('FLASK_ENV') != 'production'
    print(f"🚀  http://localhost:{PORT}  adresinde çalışıyor...")
    app.run(debug=debug, port=PORT, use_reloader=False)
