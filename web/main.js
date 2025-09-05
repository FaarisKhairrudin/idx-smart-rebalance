// Hero CTA scroll ke nominal section
document.getElementById('heroPredictBtn').onclick = function() {
    document.getElementById('nominal-section').scrollIntoView({behavior: 'smooth'});
};

// ===== Nominal IDR Section Logic =====
document.addEventListener('DOMContentLoaded', function() {
    const moneyForm = document.getElementById('moneyForm');
    const numberInput = document.getElementById('number_input');
    const nominalResult = document.getElementById('nominalResult');
    const nominalFlexContainer = document.getElementById('nominalFlexContainer');
    const moneyDisplayValue = document.getElementById('moneyDisplayValue');
    const sectorTableContainer = document.getElementById('sectorTableContainer');
    const sectorTableBody = document.getElementById('sectorTableBody');
    const pieChartContainer = document.getElementById('pieChartContainer');
    const pieChart = document.getElementById('pieChart');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const loadingText = document.getElementById('loadingText');
    const nominalSection = document.getElementById('nominal-section');

    // Variabel untuk menyimpan data hasil prediksi
    let savedPredictionData = null;
    let savedRecommendations = null;
    let isDataLoaded = false;
    let isPieChartCreated = false;

    // Data sektor dan persentase
    const sectors = [
        ['Basic Materials', 4.34],
        ['Consumer Cyclicals', 4.47],
        ['Consumer Non-Cyclicals', 6.35],
        ['Energy', 12.75],
        ['Financials', 5.10],
        ['Industrials', 12.66],
        ['Infrastuctures', 6.05],
        ['Kesehatan', 13.04],
        ['Properties Real Estate', 8.11],
        ['Technology', 4.06],
        ['Transportation Logistic', 23.07]
    ];

    // Format IDR
    function formatIDR(amount) {
        return new Intl.NumberFormat('id-ID', {
            style: 'currency',
            currency: 'IDR',
            minimumFractionDigits: 0
        }).format(amount);
    }

    // Validasi input hanya angka
    if (numberInput) {
        numberInput.addEventListener('input', function() {
            this.value = this.value.replace(/\D/g, '');
            // Mencegah input dimulai dengan 0
            if (this.value.length > 0 && this.value[0] === '0') {
                this.value = this.value.replace(/^0+/, '');
            }
        });
        
        // Tambahkan focus effect tanpa reset
        numberInput.addEventListener('focus', function() {
            this.style.transform = 'scale(1.02)';
            this.parentElement.style.boxShadow = '0 4px 12px rgba(60, 60, 60, 0.12)';
        });
        
        numberInput.addEventListener('blur', function() {
            this.style.transform = 'scale(1)';
            this.parentElement.style.boxShadow = '0 1px 3px rgba(60, 60, 60, 0.04)';
        });
    }

    // Handle form submit
    if (moneyForm) {
        moneyForm.addEventListener('submit', function(e) {
            e.preventDefault(); // Prevent page refresh
            
            const inputValue = numberInput.value.trim();
            
            // Validasi input
            if (!/^\d+$/.test(inputValue)) {
                alert('Masukkan angka saja!');
                return;
            }
            
            const nominal = parseFloat(inputValue);
            if (isNaN(nominal) || nominal <= 0) {
                alert('Nominal harus lebih dari 0!');
                return;
            }

            // Cek apakah data sudah pernah dimuat
            if (isDataLoaded && savedRecommendations) {
                console.log('Menggunakan data tersimpan, hanya update nominal...');
                // Tampilkan pesan bahwa menggunakan data tersimpan
                setLoading(true, 'Menggunakan data tersimpan...');
                setTimeout(() => {
                    setLoading(false);
                    // Hanya update nominal dan tabel tanpa prediksi ulang
                    showNominalAndTable(nominal, savedRecommendations);
                }, 800);
                return;
            }

            // LANGKAH 1: Jalankan prediksi untuk pertama kali
            const selectedSectors = getSelectedSectors();
            console.log('Sectors selected:', selectedSectors);
            setLoading(true, 'Memulai prediksi...');
            simulasiChart.innerHTML = '';
            
            fetch('http://127.0.0.1:8000/predict')
                .then(res => res.json())
                .then(data => {
                    let poll = setInterval(() => {
                        fetch('http://127.0.0.1:8000/predict/status')
                            .then(res => res.json())
                            .then(statusData => {
                                setLoading(true, statusData.status);
                                if (statusData.status === 'Selesai') {
                                    clearInterval(poll);
                                    setLoading(false);
                                    
                                    // Simpan data hasil prediksi
                                    savedPredictionData = statusData.result;
                                    savedRecommendations = statusData.result.recommendations;
                                    isDataLoaded = true;
                                    
                                    showChart(statusData.result.data, statusData.result.predictions, selectedSectors);
                                    console.log('Rekomendasi:', statusData.result.recommendations);
                                    
                                    // LANGKAH 2: Setelah prediksi selesai, tampilkan nominal dan tabel
                                    showNominalAndTable(nominal, statusData.result.recommendations);
                                    
                                } else if (statusData.status.startsWith('Error')) {
                                    clearInterval(poll);
                                    setLoading(false);
                                    alert(statusData.status);
                                }
                            })
                            .catch(() => {
                                clearInterval(poll);
                                setLoading(false);
                                alert('Gagal polling status backend.');
                            });
                    }, 1500);
                })
                .catch(() => {
                    setLoading(false);
                    alert('Tidak dapat memulai prediksi.');
                });
        });
    }
    
    // Fungsi terpisah untuk menampilkan nominal dan tabel
    function showNominalAndTable(nominal, recommendations) {
        // Smooth transition untuk hasil
        moneyDisplayValue.textContent = formatIDR(nominal);
        
        // Jika sudah ada hasil sebelumnya, update langsung tanpa animasi
        if (nominalFlexContainer.classList.contains('show-result')) {
            // Update tabel langsung
            updateSectorTable(nominal, recommendations);
            // Update pie chart
            createPieChart(recommendations, nominal);
        } else {
            // Animasi pertama kali
            nominalResult.style.display = 'flex';
            
            // Delay sedikit untuk smooth transition
            setTimeout(() => {
                nominalFlexContainer.classList.add('show-result');
            }, 50);
            
            // Tampilkan tabel setelah animasi flex selesai
            setTimeout(() => {
                updateSectorTable(nominal, recommendations);
                // Tampilkan pie chart
                createPieChart(recommendations, nominal);
                pieChartContainer.style.display = 'block';
                setTimeout(() => {
                    pieChartContainer.classList.add('show');
                }, 100);
                // Tampilkan tabel setelah pie chart
                setTimeout(() => {
                    sectorTableContainer.style.display = 'block';
                    setTimeout(() => {
                        sectorTableContainer.classList.add('show');
                    }, 100);
                }, 200);
            }, 400);
        }
    }
    
    // Fungsi untuk update tabel sektor
    function updateSectorTable(nominal, recommendations) {
        let tableHTML = '';
        
        // Debug: tampilkan struktur data recommendations
        console.log('Recommendations data:', recommendations);
        if (recommendations && recommendations.length > 0) {
            console.log('First recommendation item:', recommendations[0]);
            console.log('Keys in first item:', Object.keys(recommendations[0]));
        }
        
        // Gunakan data rekomendasi dari API jika tersedia
        if (recommendations && recommendations.length > 0) {
            // Array untuk menyimpan data yang sudah diproses
            let processedData = [];
            
            recommendations.forEach((item, index) => {
                console.log(`Item ${index}:`, item);
                
                // Coba berbagai kemungkinan nama field
                let sectorName = item.sektor || item.Sector || item.sector || item.SECTOR || 
                                  item.Sektor || item.name || item.Name || 'Unknown';
                
                // Hilangkan tanda underscore dari nama sektor
                sectorName = sectorName.replace(/_/g, ' ');
                                  
                let proportion = parseFloat(
                    item.proporsi || item.Proportion || item.proportion || item.PROPORTION ||
                    item.Proporsi || item.weight || item.Weight || item.allocation || 
                    item.Allocation || 0
                );
                
                // Bulatkan proporsi ke 4 angka di belakang koma
                proportion = Math.round(proportion * 10000) / 10000;
                
                // Jika proporsi masih 0, coba cari field persentase langsung
                let percentage;
                if (proportion > 0) {
                    percentage = proportion * 100; // Konversi ke persen setelah pembulatan
                } else {
                    // Coba cari field persentase langsung
                    const directPercentage = parseFloat(
                        item.persentase || item.Persentase || item.percentage || item.Percentage ||
                        item.PERCENTAGE || item.percent || item.Percent || 0
                    );
                    percentage = directPercentage;
                    // Jika persentase > 1, kemungkinan sudah dalam bentuk persen, jika <= 1 kemungkinan decimal
                    if (directPercentage <= 1 && directPercentage > 0) {
                        percentage = directPercentage * 100;
                    }
                }
                
                // Gunakan proporsi yang sudah dibulatkan untuk kalkulasi nominal
                const amount = nominal * proportion;
                
                console.log(`Sektor: ${sectorName}, Proportion: ${proportion}, Percentage: ${percentage}`);
                
                // Simpan data yang sudah diproses
                processedData.push({
                    sectorName: sectorName,
                    percentage: percentage,
                    amount: amount
                });
            });
            
            // Urutkan berdasarkan persentase dari terbesar ke terkecil (descending)
            processedData.sort((a, b) => b.percentage - a.percentage);
            
            // Buat HTML dari data yang sudah diurutkan
            processedData.forEach((data) => {
                tableHTML += `
                    <tr>
                        <td>${data.sectorName}</td>
                        <td>${data.percentage.toFixed(2)}%</td>
                        <td>${formatIDR(data.amount)}</td>
                    </tr>
                `;
            });
        } else {
            console.log('Using fallback static data');
            // Fallback ke data statis jika rekomendasi tidak tersedia
            sectors.forEach(([sectorName, percentage]) => {
                const amount = nominal * percentage / 100;
                tableHTML += `
                    <tr>
                        <td>${sectorName}</td>
                        <td>${percentage.toFixed(2)}%</td>
                        <td>${formatIDR(amount)}</td>
                    </tr>
                `;
            });
        }
        
        sectorTableBody.innerHTML = tableHTML;
    }
    
    // Fungsi untuk membuat pie chart
    function createPieChart(recommendations, currentNominal = 0) {
        let chartData = [];
        let colors = [
            '#0077b6', '#00b4d8', '#90e0ef', '#caf0f8', '#f77f00',
            '#fcbf49', '#f8961e', '#f3722c', '#f94144', '#f72585',
            '#7209b7'
        ];
        
        // Ambil nominal dari display jika tidak diberikan parameter
        if (!currentNominal) {
            const nominalText = moneyDisplayValue.textContent;
            const nominalNumber = nominalText.replace(/[^\d]/g, '');
            currentNominal = parseFloat(nominalNumber) || 0;
        }
        
        console.log('Current nominal for pie chart:', currentNominal);
        
        if (recommendations && recommendations.length > 0) {
            // Proses data yang sama seperti di tabel
            let processedData = [];
            
            recommendations.forEach((item) => {
                let sectorName = item.sektor || item.Sector || item.sector || item.SECTOR || 
                                  item.Sektor || item.name || item.Name || 'Unknown';
                
                // Hilangkan tanda underscore dari nama sektor
                sectorName = sectorName.replace(/_/g, ' ');
                
                let proportion = parseFloat(
                    item.proporsi || item.Proportion || item.proportion || item.PROPORTION ||
                    item.Proporsi || item.weight || item.Weight || item.allocation || 
                    item.Allocation || 0
                );
                
                // Bulatkan proporsi ke 4 angka di belakang koma
                proportion = Math.round(proportion * 10000) / 10000;
                
                let percentage;
                if (proportion > 0) {
                    percentage = proportion * 100;
                } else {
                    const directPercentage = parseFloat(
                        item.persentase || item.Persentase || item.percentage || item.Percentage ||
                        item.PERCENTAGE || item.percent || item.Percent || 0
                    );
                    percentage = directPercentage;
                    if (directPercentage <= 1 && directPercentage > 0) {
                        percentage = directPercentage * 100;
                    }
                }
                
                if (percentage > 0) {
                    processedData.push({
                        sectorName: sectorName,
                        percentage: percentage
                    });
                }
            });
            
            // Urutkan berdasarkan persentase dari terbesar ke terkecil
            processedData.sort((a, b) => b.percentage - a.percentage);
            
            // Siapkan data untuk pie chart
            chartData = [{
                values: processedData.map(item => item.percentage),
                labels: processedData.map(item => item.sectorName),
                type: 'pie',
                hole: 0.3, // Donut chart
                marker: {
                    colors: colors,
                    line: {
                        color: '#ffffff',
                        width: 3
                    }
                },
                textinfo: 'none',
                textposition: 'inside',
                textfont: {
                    size: 13,
                    color: '#000000',
                    family: 'Segoe UI, Arial, sans-serif'
                },
                insidetextorientation: 'radial',
                hovertemplate: '<b>%{label}</b>' +
                              '<br>Persentase: %{percent}' +
                              '<extra></extra>',
                rotation: 0, // Mulai dari atas
                direction: 'clockwise' // Rotasi searah jarum jam
            }];
        } else {
            // Fallback ke data statis
            chartData = [{
                values: sectors.map(item => item[1]),
                labels: sectors.map(item => item[0]),
                type: 'pie',
                hole: 0.3,
                marker: {
                    colors: colors,
                    line: {
                        color: '#ffffff',
                        width: 3
                    }
                },
                textinfo: 'none',
                textposition: 'inside',
                textfont: {
                    size: 13,
                    color: '#000000',
                    family: 'Segoe UI, Arial, sans-serif'
                },
                insidetextorientation: 'radial',
                hovertemplate: '<b>%{label}</b>' +
                              '<br>Persentase: %{percent}' +
                              '<extra></extra>',
                rotation: 0,
                direction: 'clockwise'
            }];
        }
        
        const layout = {
            showlegend: true,
            legend: {
                orientation: 'v',
                x: 1,
                y: 0.5,
                xanchor: 'left',
                yanchor: 'middle',
                font: {
                    size: 14,
                    color: '#0077b6',
                    family: 'Segoe UI, Arial, sans-serif'
                },
                bgcolor: 'rgba(255,255,255,0.95)',
                bordercolor: 'rgba(0,119,182,0.3)',
                borderwidth: 2,
                itemsizing: 'constant',
                itemwidth: 35,
                itemclick: 'toggle',
                itemdoubleclick: 'toggleothers',
                tracegroupgap: 5,
                traceorder: 'normal'
            },
            margin: {
                t: 40,
                b: 40,
                l: 40,
                r: 250,
                autoexpand: false,
                pad: 0
            },
            width: 800,
            height: 450,
            autosize: false,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {
                family: 'Segoe UI, Arial, sans-serif',
                size: 12,
                color: '#0077b6'
            },
            hoverlabel: {
                bgcolor: '#0077b6',
                bordercolor: '#ffffff',
                font: {
                    color: '#ffffff',
                    size: 12
                }
            },
            annotations: [{
                text: 'Portofolio<br>Allocation',
                x: 0.5,
                y: 0.5,
                xref: 'paper',
                yref: 'paper',
                xanchor: 'center',
                yanchor: 'middle',
                showarrow: false,
                font: {
                    size: 14,
                    color: '#0077b6',
                    family: 'Segoe UI, Arial, sans-serif'
                }
            }]
        };
        
        const config = {
            responsive: false,
            displayModeBar: false,
            doubleClick: false,
            showTips: false,
            staticPlot: false,
            scrollZoom: false,
            editable: false,
            autosizable: false,
            fillFrame: false,
            toImageButtonOptions: {
                format: 'png',
                filename: 'portfolio_distribution',
                height: 450,
                width: 800,
                scale: 1
            }
        };
        
        // Responsive legend adjustment
        const isMobile = window.innerWidth <= 700;
        if (isMobile) {
            layout.legend.orientation = 'h';
            layout.legend.x = 0.5;
            layout.legend.y = -0.1;
            layout.legend.xanchor = 'center';
            layout.legend.yanchor = 'top';
            layout.legend.font.size = 12;
            layout.margin.b = 120;
            layout.margin.r = 40;
            layout.width = 400;
            layout.height = 350;
        }
        
        // Cek apakah chart sudah pernah dibuat
        if (!isPieChartCreated) {
            // Buat pie chart pertama kali dengan animasi
            Plotly.newPlot('pieChart', chartData, layout, config).then(() => {
                isPieChartCreated = true;
                // Animasi rotasi masuk
                Plotly.animate('pieChart', {
                    data: [{
                        rotation: 360 // Rotasi penuh
                    }]
                }, {
                    transition: {
                        duration: 1500,
                        easing: 'cubic-in-out'
                    },
                    frame: {
                        duration: 1500
                    }
                }).then(() => {
                    // Reset rotasi ke posisi normal setelah animasi
                    Plotly.restyle('pieChart', {'rotation': 0});
                });
            });
        } else {
            // Update chart yang sudah ada tanpa animasi
            Plotly.react('pieChart', chartData, layout, config);
        }
        
        // Event listener untuk window resize dan zoom dengan debouncing
        let resizeTimeout;
        const handleResize = () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                const isMobileNow = window.innerWidth <= 700;
                const currentLayout = {
                    ...layout,
                    width: null,
                    height: null,
                    autosize: true
                };
                
                if (isMobileNow) {
                    currentLayout.legend = {
                        ...currentLayout.legend,
                        orientation: 'h',
                        x: 0.5,
                        y: -0.1,
                        xanchor: 'center',
                        yanchor: 'top',
                        font: { ...currentLayout.legend.font, size: 12 },
                        itemclick: 'toggle',
                        itemdoubleclick: 'toggleothers'
                    };
                    currentLayout.margin = {
                        ...currentLayout.margin,
                        b: 120,
                        r: 40,
                        l: 40,
                        t: 40
                    };
                    currentLayout.width = 400;
                    currentLayout.height = 350;
                } else {
                    currentLayout.legend = {
                        ...currentLayout.legend,
                        orientation: 'v',
                        x: 1,
                        y: 0.5,
                        xanchor: 'left',
                        yanchor: 'middle',
                        font: { ...currentLayout.legend.font, size: 14 },
                        itemclick: 'toggle',
                        itemdoubleclick: 'toggleothers'
                    };
                    currentLayout.margin = {
                        ...currentLayout.margin,
                        b: 40,
                        r: 250,
                        l: 40,
                        t: 40
                    };
                    currentLayout.width = 800;
                    currentLayout.height = 450;
                }
                
                if (isPieChartCreated) {
                    Plotly.relayout('pieChart', currentLayout);
                }
            }, 100);
        };
        
        // Hapus event listener lama jika ada, lalu tambahkan yang baru
        window.removeEventListener('resize', handleResize);
        window.addEventListener('resize', handleResize);
    }
});

// Deklarasi variabel hanya sekali
const simulasiChart = document.getElementById('simulasiChart');

function getSelectedSectors() {
    return [
        "Basic Materials",
        "Consumer Cyclicals",
        "Consumer Non-Cyclicals",
        "Energy",
        "Financials",
        "Industrials",
        "Infrastuctures",
        "Kesehatan",
        "Properties & Real Estate",
        "Technology",
        "Transportation & Logistic"
    ];
}

function setLoading(isLoading, statusText = 'Memulai prediksi...') {
    const loadingOverlay = document.getElementById('loadingOverlay');
    const loadingText = document.getElementById('loadingText');
    const nominalSection = document.getElementById('nominal-section');
    
    if (isLoading) {
        // Tampilkan overlay loading
        loadingOverlay.style.display = 'flex';
        setTimeout(() => {
            loadingOverlay.classList.add('show');
        }, 10);
        
        // Disable section
        nominalSection.classList.add('loading-disabled');
        
        // Update text
        loadingText.textContent = statusText;
    } else {
        // Sembunyikan overlay loading
        loadingOverlay.classList.remove('show');
        setTimeout(() => {
            loadingOverlay.style.display = 'none';
        }, 400);
        
        // Enable section
        nominalSection.classList.remove('loading-disabled');
    }
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

if (window.location.hash === '#nominal-section') {
    setTimeout(() => {
        document.getElementById('nominal-section').scrollIntoView({behavior: 'smooth'});
    }, 500);
}
