window.onload = function() {
    // Prepare the histogram for the GUI

    const scriptTag = document.getElementById('drawHistogram');
    const dataBlue = JSON.parse(scriptTag.getAttribute('histogram_blue'));
    const dataGreen = JSON.parse(scriptTag.getAttribute('histogram_green'));
    const dataRed = JSON.parse(scriptTag.getAttribute('histogram_red'));

    const canvas = document.getElementById('rgbHistogramCanvas');
    const ctx = canvas.getContext('2d');

    if (!canvas) {
        console.error("Canvas non trovato");
        return;
    }

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Array.from({ length: 256 }, (_, i) => i),
            datasets: [
                {
                    label: 'Blue',
                    data: dataBlue,
                    backgroundColor: 'rgba(0, 0, 255, 1)'
                },
                {
                    label: 'Green',
                    data: dataGreen,
                    backgroundColor: 'rgba(0, 255, 0, 1)'
                },
                {
                    label: 'Red',
                    data: dataRed,
                    backgroundColor: 'rgba(255, 0, 0, 1)'
                }
            ]
        },
        options: {
            scales: {
                x: {

                },
                y: {
                    beginAtZero: true,
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'RGB Histogram of the Image',
                    font: {
                        size: 18,
                        weight: 'bold'
                    },
                    color: 'blue',
                    padding: {
                        top: 10,
                        bottom: 30
                    }
                }
            }
        }
    });
};