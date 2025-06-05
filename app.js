// app.js
document.addEventListener('DOMContentLoaded', () => {
    const tmdMassSlider = document.getElementById('tmd_mass');
    const tmdMassValue = document.getElementById('tmd_mass_value');
    const tmdStiffnessSlider = document.getElementById('tmd_stiffness');
    const tmdStiffnessValue = document.getElementById('tmd_stiffness_value');
    const tmdDampingSlider = document.getElementById('tmd_damping');
    const tmdDampingValue = document.getElementById('tmd_damping_value');
    const useTmdCheckbox = document.getElementById('use_tmd_checkbox');

    const runSimButton = document.getElementById('run_simulation_button');
    const loadOptimalButton = document.getElementById('load_optimal_button');
    
    const statusMessage = document.getElementById('status_message');

    const animCanvas = document.getElementById('building_animation_canvas');
    const animCtx = animCanvas.getContext('2d');
    const plotCanvas = document.getElementById('displacement_plot_canvas');
    let displacementChart = null; // Untuk menyimpan instance Chart.js

    // Update nilai slider secara dinamis
    tmdMassSlider.oninput = () => tmdMassValue.textContent = tmdMassSlider.value;
    tmdStiffnessSlider.oninput = () => tmdStiffnessValue.textContent = tmdStiffnessSlider.value;
    tmdDampingSlider.oninput = () => tmdDampingValue.textContent = tmdDampingSlider.value;

    // Fungsi untuk menjalankan simulasi
    async function runSimulation() {
        statusMessage.textContent = 'Menjalankan simulasi...';
        const tmdParams = {
            mass_kg: parseFloat(tmdMassSlider.value),
            stiffness_N_m: parseFloat(tmdStiffnessSlider.value),
            damping_Ns_m: parseFloat(tmdDampingSlider.value)
        };
        const useTmd = useTmdCheckbox.checked;

        // Parameter gedung (bisa dibuat dinamis jika ada slidernya)
        const buildingParams = {
            num_stories: 3 // Sesuai default di backend Flask
            // Jika ada slider gedung, ambil nilainya di sini
        };

        try {
            const response = await fetch('http://127.0.0.1:5000/simulate_html', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    tmd_params: tmdParams, 
                    use_tmd: useTmd,
                    // building_params: buildingParams // Jika dikirim dari frontend
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error |
| `HTTP error! status: ${response.status}`);
            }

            const results = await response.json();
            statusMessage.textContent = 'Simulasi selesai. Memproses hasil...';
            
            updateVisualizations(results.time, results.displacements_building, results.displacement_tmd_relative);
            statusMessage.textContent = 'Visualisasi diperbarui.';

        } catch (error) {
            console.error('Error saat simulasi:', error);
            statusMessage.textContent = `Error: ${error.message}`;
        }
    }

    runSimButton.addEventListener('click', runSimulation);

    loadOptimalButton.addEventListener('click', async () => {
        statusMessage.textContent = 'Memuat parameter TMD optimal...';
        try {
            const response = await fetch('http://127.0.0.1:5000/get_optimal_tmd');
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error |
| `HTTP error! status: ${response.status}`);
            }
            const optimalParams = await response.json();

            tmdMassSlider.value = optimalParams.mass_kg;
            tmdStiffnessSlider.value = optimalParams.stiffness_N_m;
            tmdDampingSlider.value = optimalParams.damping_Ns_m;
            
            // Update tampilan nilai slider
            tmdMassValue.textContent = optimalParams.mass_kg;
            tmdStiffnessValue.textContent = optimalParams.stiffness_N_m;
            tmdDampingValue.textContent = optimalParams.damping_Ns_m;
            
            useTmdCheckbox.checked = true; // Asumsi optimal selalu dengan TMD
            statusMessage.textContent = 'Parameter optimal dimuat. Jalankan simulasi untuk melihat hasilnya.';
            // Otomatis jalankan simulasi setelah memuat parameter optimal
            await runSimulation();

        } catch (error) {
            console.error('Error memuat parameter optimal:', error);
            statusMessage.textContent = `Error: ${error.message}`;
        }
    });

    function updateVisualizations(time, buildingDisplacements, tmdDisplacementRelative) {
        // 1. Update Plot Respons
        const topFloorDisplacement = buildingDisplacements.map(floorDisps => floorDisps); // Lantai teratas

        if (displacementChart) {
            displacementChart.destroy(); // Hancurkan chart lama sebelum membuat yang baru
        }
        
        const datasets =, y: disp})),
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1,
            fill: false
        }];

        if (tmdDisplacementRelative && useTmdCheckbox.checked) {
            datasets.push({
                label: 'Perpindahan Relatif TMD (m)',
                data: tmdDisplacementRelative.map((disp, i) => ({x: time[i], y: disp})),
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1,
                fill: false,
                borderDash:  // Garis putus-putus
            });
        }

        displacementChart = new Chart(plotCanvas, {
            type: 'line',
            data: {
                // labels: time, // Tidak perlu jika data adalah {x,y}
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear', // Atau 'time' jika format waktu sesuai
                        title: { display: true, text: 'Waktu (s)' }
                    },
                    y: {
                        title: { display: true, text: 'Perpindahan (m)' }
                    }
                },
                animation: { duration: 0 } // Matikan animasi Chart.js untuk update cepat
            }
        });

        // 2. Update Animasi Gedung (fungsi sederhana)
        // Asumsi gedung 3 lantai dari DEFAULT_BUILDING_PROPS
        const numFloors = buildingDisplacements.length; 
        const floorHeight = animCanvas.height / (numFloors + 2); // +2 untuk dasar dan ruang atas
        const buildingWidth = animCanvas.width * 0.3;
        const buildingBaseX = (animCanvas.width - buildingWidth) / 2;
        let currentFrame = 0;
        const scaleFactor = 500; // Skala untuk visualisasi perpindahan (sesuaikan agar terlihat)

        function animateBuilding() {
            if (currentFrame >= time.length) {
                currentFrame = 0; // Loop animasi atau stop
                // return; 
            }

            animCtx.clearRect(0, 0, animCanvas.width, animCanvas.height);
            
            // Gambar dasar
            animCtx.fillStyle = 'gray';
            animCtx.fillRect(0, animCanvas.height - floorHeight, animCanvas.width, floorHeight);

            let lastY = animCanvas.height - floorHeight;
            let lastX = buildingBaseX;

            // Gambar lantai-lantai gedung
            for (let i = 0; i < numFloors; i++) {
                const dispX = buildingDisplacements[currentFrame][i] * scaleFactor;
                const currentFloorY = animCanvas.height - floorHeight * (i + 2);
                
                animCtx.fillStyle = 'lightblue';
                animCtx.fillRect(buildingBaseX + dispX, currentFloorY, buildingWidth, floorHeight * 0.8);
                
                // Gambar kolom (sederhana)
                animCtx.strokeStyle = 'black';
                animCtx.lineWidth = 2;
                animCtx.beginPath();
                animCtx.moveTo(lastX + buildingWidth/4, lastY);
                animCtx.lineTo(buildingBaseX + dispX + buildingWidth/4, currentFloorY + floorHeight*0.8);
                animCtx.moveTo(lastX + (3*buildingWidth)/4, lastY);
                animCtx.lineTo(buildingBaseX + dispX + (3*buildingWidth)/4, currentFloorY + floorHeight*0.8);
                animCtx.stroke();

                lastY = currentFloorY;
                lastX = buildingBaseX + dispX;

                // Jika ini lantai teratas dan TMD aktif
                if (i === numFloors - 1 && useTmdCheckbox.checked && tmdDisplacementRelative) {
                    const tmdDispX = (buildingDisplacements[currentFrame][i] + tmdDisplacementRelative[currentFrame]) * scaleFactor;
                    const tmdY = currentFloorY - floorHeight * 0.5;
                    const tmdWidth = buildingWidth * 0.3;
                    const tmdHeight = floorHeight * 0.4;
                    
                    animCtx.fillStyle = 'red';
                    animCtx.fillRect(buildingBaseX - tmdWidth/2 + tmdDispX + buildingWidth/2, tmdY, tmdWidth, tmdHeight);
                    
                    // "Pegas" TMD (garis sederhana)
                    animCtx.beginPath();
                    animCtx.moveTo(buildingBaseX + dispX + buildingWidth/2, currentFloorY);
                    animCtx.lineTo(buildingBaseX - tmdWidth/2 + tmdDispX + buildingWidth/2 + tmdWidth/2, tmdY + tmdHeight/2);
                    animCtx.stroke();
                }
            }
            currentFrame = (currentFrame + 5) % time.length; // Loncat frame agar animasi tidak terlalu lambat
                                                            // atau sesuaikan dengan dt dan refresh rate
            requestAnimationFrame(animateBuilding);
        }
        
        // Hentikan animasi lama jika ada
        if (window.animationFrameId) {
            cancelAnimationFrame(window.animationFrameId);
        }
        window.animationFrameId = requestAnimationFrame(animateBuilding); // Mulai animasi baru
    }
    
    // Jalankan simulasi awal dengan parameter default (tanpa TMD)
    useTmdCheckbox.checked = false;
    runSimulation(); 
});