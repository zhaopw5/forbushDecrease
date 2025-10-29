#include <bits/stdc++.h>
#include <filesystem>
// 移除 matplotlibcpp 头文件与命名空间
// #include "matplotlibcpp.h"
// namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

// 引入 ROOT 头文件
#include "TCanvas.h"
#include "TPad.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TAxis.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TDatime.h"  // 新增：用于设置 UNIX epoch

using namespace std;

struct OmniRow {
    int64_t ts;     // seconds since epoch (UTC)
    double Bz_gse;
    double V;
};

struct FluxRow {
    int64_t ts;     // seconds since epoch (UTC)
    double flux;
    double err;
};

static inline optional<int64_t> parse_ts(const string& s) {
    // Accept "YYYY-mm-dd HH:MM:SS" or "YYYY-mm-dd"
    std::tm tm{}; tm.tm_isdst = 0;
    std::istringstream ss(s);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    if (ss.fail()) {
        ss.clear(); ss.str(s);
        ss >> std::get_time(&tm, "%Y-%m-%d");
        if (ss.fail()) return nullopt;
    }
    // Use timegm for UTC
    #if defined(__USE_XOPEN2K8) || defined(__GLIBC__) || defined(__APPLE__)
    time_t t = timegm(&tm);
    #else
    time_t t = mktime(&tm); // fallback: local time (may shift vs UTC)
    #endif
    if (t == (time_t)-1) return nullopt;
    return static_cast<int64_t>(t);
}

static inline int64_t floor_day(int64_t ts) {
    time_t t = ts;
    std::tm* utc = gmtime(&t);
    std::tm tm = *utc;
    tm.tm_hour = 0; tm.tm_min = 0; tm.tm_sec = 0;
    #if defined(__USE_XOPEN2K8) || defined(__GLIBC__) || defined(__APPLE__)
    return static_cast<int64_t>(timegm(&tm));
    #else
    return static_cast<int64_t>(mktime(&tm));
    #endif
}

static inline int64_t floor_week(int64_t ts) {
    time_t t = ts;
    std::tm* utc = gmtime(&t);
    std::tm tm = *utc;
    // tm_wday: 0=Sunday ... 6=Saturday; we choose Monday as week start
    int wday = tm.tm_wday;
    int days_from_monday = (wday + 6) % 7; // 0 for Monday
    // Go back to Monday 00:00:00
    tm.tm_mday -= days_from_monday;
    tm.tm_hour = 0; tm.tm_min = 0; tm.tm_sec = 0;
    #if defined(__USE_XOPEN2K8) || defined(__GLIBC__) || defined(__APPLE__)
    return static_cast<int64_t>(timegm(&tm));
    #else
    return static_cast<int64_t>(mktime(&tm));
    #endif
}

static inline string format_label_MdY(int64_t ts) {
    // "Month Day\nYear", e.g., "March 17\n2015"
    time_t t = ts;
    char buf[64];
    std::tm* utc = gmtime(&t);
    strftime(buf, sizeof(buf), "%B %d\n%Y", utc);
    return string(buf);
}

static vector<string> split_csv_line(const string& line) {
    // Simple CSV split by comma; not handling quoted commas for simplicity
    vector<string> out;
    string cur;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

static inline double stod_safe(const string& s) {
    if (s.empty()) return NAN;
    try { return stod(s); } catch (...) { return NAN; }
}

struct Settings {
    string OMNI_HOURLY_CSV = "/home/zpw/Files/forbushDecrease/solardata/omni_min2015_hourly.csv";
    string FLUX_LONG_CSV   = "/home/zpw/Files/forbushDecrease/hourlyAMS/flux_long.csv";
    double RIGIDITY_MIN = 1.51;
    double RIGIDITY_MAX = 1.71;
    optional<int64_t> TIME_MIN = parse_ts("2015-03-10 00:00:00");
    optional<int64_t> TIME_MAX = parse_ts("2015-03-24 23:00:00");
    string RESAMPLE = ""; // "", "D", "W"
    string OUT_DIR  = "plots";
    string OUT_NAME; // computed
};

static void read_omni(const string& path, vector<OmniRow>& out) {
    ifstream fin(path);
    if (!fin) throw runtime_error("Cannot open: " + path);
    string header;
    if (!getline(fin, header)) throw runtime_error("Empty file: " + path);
    auto cols = split_csv_line(header);
    unordered_map<string, int> idx;
    for (int i = 0; i < (int)cols.size(); ++i) idx[cols[i]] = i;
    vector<string> needed = {"datetime","Bz_gse","V"};
    for (auto& k: needed) if (!idx.count(k)) throw runtime_error("Missing column '"+k+"' in "+path);

    string line;
    out.clear();
    while (getline(fin, line)) {
        if (line.empty()) continue;
        auto toks = split_csv_line(line);
        if ((int)toks.size() < (int)cols.size()) continue;
        auto ts_opt = parse_ts(toks[idx["datetime"]]);
        if (!ts_opt) continue;
        double bz = stod_safe(toks[idx["Bz_gse"]]);
        double v  = stod_safe(toks[idx["V"]]);
        if (isnan(bz) || isnan(v)) continue;
        out.push_back({*ts_opt, bz, v});
    }
    sort(out.begin(), out.end(), [](auto& a, auto& b){ return a.ts < b.ts; });
}

static void read_flux_bin(const string& path, double RMIN, double RMAX, vector<FluxRow>& out) {
    ifstream fin(path);
    if (!fin) throw runtime_error("Cannot open: " + path);
    string header;
    if (!getline(fin, header)) throw runtime_error("Empty file: " + path);
    auto cols = split_csv_line(header);
    unordered_map<string, int> idx;
    for (int i = 0; i < (int)cols.size(); ++i) idx[cols[i]] = i;
    vector<string> needed = {"date","rigidity_min","rigidity_max","flux","error_bar"};
    for (auto& k: needed) if (!idx.count(k)) throw runtime_error("Missing column '"+k+"' in "+path);

    const double tol = 1e-8;
    string line;
    out.clear();
    while (getline(fin, line)) {
        if (line.empty()) continue;
        auto toks = split_csv_line(line);
        if ((int)toks.size() < (int)cols.size()) continue;
        double rmin = stod_safe(toks[idx["rigidity_min"]]);
        double rmax = stod_safe(toks[idx["rigidity_max"]]);
        if (fabs(rmin - RMIN) > tol || fabs(rmax - RMAX) > tol) continue;

        auto ts_opt = parse_ts(toks[idx["date"]]);
        if (!ts_opt) continue;
        double f = stod_safe(toks[idx["flux"]]);
        double e = stod_safe(toks[idx["error_bar"]]);
        if (isnan(f) || isnan(e)) continue;
        out.push_back({*ts_opt, f, e});
    }
    if (out.empty()) {
        ostringstream oss;
        oss << "No rows for bin [" << RMIN << ", " << RMAX << "] GV in " << path;
        throw runtime_error(oss.str());
    }
    sort(out.begin(), out.end(), [](auto& a, auto& b){ return a.ts < b.ts; });
}

template <class T>
static void apply_time_window(vector<T>& v, optional<int64_t> tmin, optional<int64_t> tmax) {
    vector<T> tmp;
    tmp.reserve(v.size());
    for (auto& row: v) {
        if (tmin && row.ts < *tmin) continue;
        if (tmax && row.ts > *tmax) continue;
        tmp.push_back(row);
    }
    v.swap(tmp);
}

static void resample_mean_omni(const vector<OmniRow>& in, const string& mode, vector<OmniRow>& out) {
    if (mode.empty()) { out = in; return; }
    map<int64_t, pair<double,int>> agg_bz; // ts_bin -> (sum, count)
    map<int64_t, pair<double,int>> agg_v;
    map<int64_t, int> counter;

    auto bin = [&](int64_t ts){
        if (mode=="D") return floor_day(ts);
        if (mode=="W") return floor_week(ts);
        return ts;
    };
    for (auto& r: in) {
        auto b = bin(r.ts);
        agg_bz[b].first += r.Bz_gse; agg_bz[b].second += 1;
        agg_v[b].first  += r.V;      agg_v[b].second  += 1;
        counter[b] += 1;
    }
    out.clear();
    out.reserve(counter.size());
    for (auto& [k, c]: counter) {
        double bz = agg_bz[k].first / max(1, agg_bz[k].second);
        double v  = agg_v[k].first  / max(1, agg_v[k].second);
        out.push_back({k, bz, v});
    }
    sort(out.begin(), out.end(), [](auto& a, auto& b){ return a.ts < b.ts; });
}

static void resample_mean_flux(const vector<FluxRow>& in, const string& mode, vector<FluxRow>& out) {
    if (mode.empty()) { out = in; return; }
    struct Agg { double sum_f=0, sum_e=0; int n=0; };
    map<int64_t, Agg> agg;
    auto bin = [&](int64_t ts){
        if (mode=="D") return floor_day(ts);
        if (mode=="W") return floor_week(ts);
        return ts;
    };
    for (auto& r: in) {
        auto b = bin(r.ts);
        auto& a = agg[b];
        a.sum_f += r.flux;
        a.sum_e += r.err;
        a.n += 1;
    }
    out.clear();
    out.reserve(agg.size());
    for (auto& [k, a]: agg) {
        out.push_back({k, a.sum_f / max(1,a.n), a.sum_e / max(1,a.n)});
    }
    sort(out.begin(), out.end(), [](auto& a, auto& b){ return a.ts < b.ts; });
}

static void intersection_align(
    const vector<FluxRow>& f,
    const vector<OmniRow>& o,
    vector<FluxRow>& f_aligned,
    vector<OmniRow>& o_aligned
) {
    unordered_set<int64_t> st;
    st.reserve(o.size()*2+1);
    for (auto& r: o) st.insert(r.ts);
    f_aligned.clear(); o_aligned.clear();
    for (auto& fr: f) {
        if (st.count(fr.ts)) f_aligned.push_back(fr);
    }
    if (f_aligned.empty()) {
        return; // no overlap
    }
    unordered_set<int64_t> keep;
    keep.reserve(f_aligned.size()*2+1);
    for (auto& fr: f_aligned) keep.insert(fr.ts);
    for (auto& orow: o) if (keep.count(orow.ts)) o_aligned.push_back(orow);
    // both are sorted by ts already
}

static void to_xy(const vector<pair<int64_t,double>>& v, vector<double>& x, vector<double>& y) {
    x.resize(v.size()); y.resize(v.size());
    for (size_t i=0;i<v.size();++i){ x[i]=static_cast<double>(v[i].first); y[i]=v[i].second; }
}

static vector<pair<int64_t,double>> mk_series(const vector<OmniRow>& v, bool bz) {
    vector<pair<int64_t,double>> s; s.reserve(v.size());
    for (auto& r: v) s.push_back({r.ts, bz ? r.Bz_gse : r.V});
    return s;
}

static vector<pair<int64_t,double>> mk_series_flux(const vector<FluxRow>& v, bool err=false) {
    vector<pair<int64_t,double>> s; s.reserve(v.size());
    for (auto& r: v) s.push_back({r.ts, err ? r.err : r.flux});
    return s;
}

static void compute_xticks_labels(int64_t xmin, int64_t xmax, int nticks,
                                  vector<double>& ticks, vector<string>& labels) {
    if (xmax <= xmin) { ticks = {double(xmin)}; labels = {format_label_MdY(xmin)}; return; }
    nticks = max(2, nticks);
    ticks.resize(nticks);
    labels.resize(nticks);
    double dx = double(xmax - xmin) / double(nticks - 1);
    for (int i = 0; i < nticks; ++i) {
        double xv = double(xmin) + dx * i;
        ticks[i] = xv;
        labels[i] = format_label_MdY(int64_t(xv));
    }
}

// 将原 main 主体逻辑移入实现函数，供 ROOT 与 main 复用
int protonFlux_solarParameters_impl() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    try {
        Settings S;
        {
            ostringstream oss;
            oss << "flux_Bz_V_" << S.RIGIDITY_MIN << "-" << S.RIGIDITY_MAX << "GV";
            S.OUT_NAME = oss.str();
        }

        vector<OmniRow> omni_raw;
        vector<FluxRow> flux_raw;
        read_omni(S.OMNI_HOURLY_CSV, omni_raw);
        read_flux_bin(S.FLUX_LONG_CSV, S.RIGIDITY_MIN, S.RIGIDITY_MAX, flux_raw);

        if (S.TIME_MIN || S.TIME_MAX) {
            apply_time_window(omni_raw, S.TIME_MIN, S.TIME_MAX);
            apply_time_window(flux_raw, S.TIME_MIN, S.TIME_MAX);
        }
        if (omni_raw.empty() || flux_raw.empty()) {
            throw runtime_error("No data to plot after applying the time window.");
        }

        vector<OmniRow> omni_plot;
        vector<FluxRow> flux_plot;
        resample_mean_omni(omni_raw, S.RESAMPLE, omni_plot);
        resample_mean_flux(flux_raw, S.RESAMPLE, flux_plot);

        // Try align by intersection
        vector<OmniRow> omni_al;
        vector<FluxRow> flux_al;
        intersection_align(flux_plot, omni_plot, flux_al, omni_al);

        bool aligned = !flux_al.empty();
        const auto& f_use = aligned ? flux_al : flux_plot;
        const auto& o_use = aligned ? omni_al : omni_plot;

        // Build series
        auto flux_s  = mk_series_flux(f_use, false);
        auto err_s   = mk_series_flux(f_use, true);
        auto bz_s    = mk_series(o_use, true);
        auto v_s     = mk_series(o_use, false);

        vector<double> xf, yf, yerr;
        vector<double> xb, yb;
        vector<double> xv, yv;

        xf.resize(flux_s.size());
        yf.resize(flux_s.size());
        yerr.resize(err_s.size());
        for (size_t i=0;i<flux_s.size();++i){
            xf[i] = double(flux_s[i].first);
            yf[i] = flux_s[i].second;
            yerr[i] = err_s[i].second;
        }
        xb.resize(bz_s.size()); yb.resize(bz_s.size());
        for (size_t i=0;i<bz_s.size();++i){ xb[i]=double(bz_s[i].first); yb[i]=bz_s[i].second; }
        xv.resize(v_s.size()); yv.resize(v_s.size());
        for (size_t i=0;i<v_s.size();++i){ xv[i]=double(v_s[i].first); yv[i]=v_s[i].second; }

        // Determine xticks range on bottom axis
        int64_t xmin = 0, xmax = 0;
        auto set_minmax = [&](const vector<double>& x){
            if (x.empty()) return;
            int64_t mn = (int64_t)*min_element(x.begin(), x.end());
            int64_t mx = (int64_t)*max_element(x.begin(), x.end());
            if (xmin==0 && xmax==0) { xmin=mn; xmax=mx; }
            else { xmin=min(xmin,mn); xmax=max(xmax,mx); }
        };
        set_minmax(xf); set_minmax(xb); set_minmax(xv);

        // ========== 使用 ROOT 绘图 ==========
        gStyle->SetOptStat(0);
        // 修正：ROOT 6.32 仅接受单参，使用 UNIX epoch 作为时间偏移
        TDatime off(1970, 1, 1, 0, 0, 0);
        gStyle->SetTimeOffset(off.Convert());

        TCanvas c("c", "flux_Bz_V", 1500, 1200);
        // 三个无缝上下堆叠的 pad
        TPad p1("p1","",0.0,0.66,1.0,1.0);
        TPad p2("p2","",0.0,0.33,1.0,0.66);
        TPad p3("p3","",0.0,0.00,1.0,0.33);

        for (TPad* p: {&p1,&p2,&p3}) {
            p->SetLeftMargin(0.10);
            p->SetRightMargin(0.03);
            p->SetGridx(true);
            p->SetGridy(true);
        }
        p1.SetTopMargin(0.06); p1.SetBottomMargin(0.02);
        p2.SetTopMargin(0.02); p2.SetBottomMargin(0.02);
        p3.SetTopMargin(0.02); p3.SetBottomMargin(0.14);

        p1.Draw(); p2.Draw(); p3.Draw();

        const int nF = (int)xf.size();
        const int nB = (int)xb.size();
        const int nV = (int)xv.size();

        // Row 1: flux ± error (TGraphErrors)
        p1.cd();
        vector<double> ex(nF, 0.0);
        TGraphErrors geF(nF, nF? &xf[0]:nullptr, nF? &yf[0]:nullptr, nF? &ex[0]:nullptr, nF? &yerr[0]:nullptr);
        geF.SetTitle("");
        geF.SetLineWidth(2);
        geF.SetMarkerStyle(20);
        geF.SetMarkerSize(0.7);
        geF.GetXaxis()->SetLimits((double)xmin, (double)xmax);
        geF.GetXaxis()->SetTimeDisplay(1);
        geF.GetXaxis()->SetTimeFormat("%b %d %Y");
        geF.GetXaxis()->SetLabelSize(0);    // 上两行隐藏 X 轴标签
        geF.GetXaxis()->SetTitleSize(0);
        geF.GetYaxis()->SetTitle("Proton Flux");
        geF.Draw("AP"); // 轴+点
        {
            auto leg = new TLegend(0.62, 0.78, 0.98, 0.94);
            std::ostringstream oss; oss << " [" << S.RIGIDITY_MIN << "-" << S.RIGIDITY_MAX << "] GV";
            leg->AddEntry(&geF, oss.str().c_str(), "lp");
            leg->SetBorderSize(0);
            leg->SetFillStyle(0);
            leg->Draw();
        }

        // Row 2: Bz_gse (TGraph)
        p2.cd();
        TGraph gB(nB, nB? &xb[0]:nullptr, nB? &yb[0]:nullptr);
        gB.SetTitle("");
        gB.SetMarkerStyle(20);
        gB.SetMarkerSize(0.6);
        gB.SetLineWidth(1);
        gB.GetXaxis()->SetLimits((double)xmin, (double)xmax);
        gB.GetXaxis()->SetTimeDisplay(1);
        gB.GetXaxis()->SetTimeFormat("%b %d %Y");
        gB.GetXaxis()->SetLabelSize(0);     // 隐藏 X 轴标签
        gB.GetXaxis()->SetTitleSize(0);
        gB.GetYaxis()->SetTitle("Bz_gse (nT)");
        gB.Draw("AP");
        {
            auto leg = new TLegend(0.62, 0.78, 0.98, 0.94);
            leg->AddEntry(&gB, "Bz_gse (Field magnitude)", "p");
            leg->SetBorderSize(0);
            leg->SetFillStyle(0);
            leg->Draw();
        }

        // Row 3: V (TGraph)
        p3.cd();
        TGraph gV(nV, nV? &xv[0]:nullptr, nV? &yv[0]:nullptr);
        gV.SetTitle("");
        gV.SetMarkerStyle(20);
        gV.SetMarkerSize(0.6);
        gV.SetLineColor(kRed+1);
        gV.SetLineWidth(1);
        gV.GetXaxis()->SetLimits((double)xmin, (double)xmax);
        gV.GetXaxis()->SetTimeDisplay(1);
        gV.GetXaxis()->SetTimeFormat("%b %d %Y"); // 若需换行，可改用更紧凑格式
        gV.GetXaxis()->SetNdivisions(510);
        gV.GetXaxis()->SetLabelSize(0.04);
        gV.GetXaxis()->SetTitleSize(0.045);
        gV.GetYaxis()->SetTitle("V (km/s)");
        gV.Draw("AP");
        {
            auto leg = new TLegend(0.62, 0.78, 0.98, 0.94);
            leg->AddEntry(&gV, "V (Flow speed)", "p");
            leg->SetBorderSize(0);
            leg->SetFillStyle(0);
            leg->Draw();
        }

        // 保存
        fs::create_directories(S.OUT_DIR);
        string suffix = S.RESAMPLE.empty() ? "" : "_" + S.RESAMPLE;
        string base = S.OUT_DIR + "/" + S.OUT_NAME + suffix;
        c.SaveAs((base + ".png").c_str());
        c.SaveAs((base + ".pdf").c_str());
        cout << "Saved:\n  " << base << ".png\n  " << base << ".pdf\n";
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

// ROOT 宏同名入口：root protonFlux_solarParameters.cpp 时会尝试调用该函数
void protonFlux_solarParameters() {
    (void)protonFlux_solarParameters_impl();
}

// 仅在非 ROOT 解释执行时编译 main，便于 g++ 正常构建和命令行运行
#ifndef __CLING__
int main() {
    return protonFlux_solarParameters_impl();
}
#endif
