// Hero CTA scroll ke simulasi
document.getElementById('heroPredictBtn').onclick = function() {
    document.getElementById('simulasi').scrollIntoView({behavior: 'smooth'});
};

// Deklarasi variabel hanya sekali
const sektorCheckboxes = document.getElementById('sektorCheckboxes');
const simulasiPredictBtn = document.getElementById('simulasiPredictBtn');
const simulasiLoader = document.getElementById('simulasiLoader');
const simulasiChart = document.getElementById('simulasiChart');
const resetBtn = document.getElementById('resetBtn');
const selectAllBtn = document.getElementById('selectAllBtn');
const simulasiStatus = document.getElementById('simulasiStatus');

// Tombol pilih semua sektor
selectAllBtn.addEventListener('click', function() {
    sektorCheckboxes.querySelectorAll('input[type="checkbox"]').forEach(box => box.checked = true);
});

function getSelectedSectors() {
    return Array.from(sektorCheckboxes.querySelectorAll('input[type="checkbox"]:checked')).map(box => box.value);
}

function setLoading(isLoading) {
    simulasiLoader.style.display = isLoading ? 'inline-block' : 'none';
    simulasiPredictBtn.disabled = isLoading;
    resetBtn.disabled = isLoading;
    sektorCheckboxes.querySelectorAll('input[type="checkbox"]').forEach(box => box.disabled = isLoading);
    simulasiStatus.textContent = isLoading ? 'Memulai prediksi...' : '';
}

function resetSimulasi() {
    sektorCheckboxes.querySelectorAll('input[type="checkbox"]').forEach(box => box.checked = false);
    simulasiChart.innerHTML = '';
    simulasiStatus.textContent = '';
    resetBtn.style.display = 'none';
    simulasiPredictBtn.style.display = 'block';
    simulasiPredictBtn.disabled = false;
    sektorCheckboxes.querySelectorAll('input[type="checkbox"]').forEach(box => box.disabled = false);
}

function showChart(data, predictions, sectors) {
    simulasiChart.innerHTML = '';
    if (!sectors.length) return;
    let chartHtml = '';
    sectors.forEach(sektor => {
        let hist = data.filter(d => d.Sector === sektor);
        let pred = predictions.filter(d => d.Sector === sektor);
        hist = hist.slice(-20);
        let traces = [];
        traces.push({
            x: hist.map(d => d.Date),
            y: hist.map(d => d.SectorVolatility_7d),
            name: sektor + ' (Hist)',
            mode: 'lines+markers',
            line: {color: '#0077b6'},
        });
        if (pred.length > 0) {
            traces.push({
                x: pred.map(d => d.Date),
                y: pred.map(d => d.SectorVolatility_7d),
                name: sektor + ' (Prediksi)',
                mode: 'lines+markers',
                line: {dash: 'dot', color: '#f77f00'}, // warna prediksi lebih kontras
                marker: {color: '#f77f00'},
            });
        }
        // Buat div unik untuk setiap sektor
        const chartId = 'chart_' + sektor.replace(/\s+/g, '_');
        chartHtml += `<div class="sektor-chart-block"><h3>Plot Volatilitas: ${sektor}</h3><div id="${chartId}" class="sektor-chart"></div></div>`;
        setTimeout(() => {
            if (traces.length === 0) {
                document.getElementById(chartId).innerHTML = '<div style="color:#888;">Tidak ada data untuk sektor terpilih.</div>';
            } else {
                Plotly.newPlot(chartId, traces, {
                    title: '',
                    xaxis: {title: 'Tanggal'},
                    yaxis: {title: 'Volatilitas'}, // label y diganti
                    legend: {orientation: 'h'},
                    margin: {t:20, l:40, r:20, b:40},
                }, {responsive:true});
            }
        }, 100);
    });
    simulasiChart.innerHTML = chartHtml;
}

simulasiPredictBtn.addEventListener('click', function() {
    const sectors = getSelectedSectors();
    if (!sectors.length) {
        alert('Pilih minimal satu sektor!');
        return;
    }
    setLoading(true);
    simulasiChart.innerHTML = '';
    simulasiStatus.textContent = 'Memulai prediksi...';
    fetch('http://127.0.0.1:8000/predict')
        .then(res => res.json())
        .then(data => {
            let poll = setInterval(() => {
                fetch('http://127.0.0.1:8000/predict/status')
                    .then(res => res.json())
                    .then(statusData => {
                        simulasiStatus.textContent = statusData.status;
                        if (statusData.status === 'Selesai') {
                            clearInterval(poll);
                            setLoading(false);
                            resetBtn.style.display = 'block';
                            simulasiPredictBtn.style.display = 'none';
                            showChart(statusData.result.data, statusData.result.predictions, sectors);
                        } else if (statusData.status.startsWith('Error')) {
                            clearInterval(poll);
                            setLoading(false);
                            simulasiStatus.textContent = statusData.status;
                            alert(statusData.status);
                        }
                    })
                    .catch(() => {
                        clearInterval(poll);
                        setLoading(false);
                        simulasiStatus.textContent = 'Gagal polling status backend.';
                        alert('Gagal polling status backend.');
                    });
            }, 1500);
        })
        .catch(() => {
            setLoading(false);
            simulasiStatus.textContent = 'Tidak dapat memulai prediksi.';
            alert('Tidak dapat memulai prediksi.');
        });
});

resetBtn.addEventListener('click', function() {
    resetSimulasi();
});

if (window.location.hash === '#simulasi') {
    setTimeout(() => {
        document.getElementById('simulasi').scrollIntoView({behavior: 'smooth'});
    }, 500);
}
