/**
 * Customer Segmentation Dashboard - Charts
 * Handles all Chart.js visualizations
 */

// ============================================
// Chart Configuration
// ============================================

// Chart.js default configuration
Chart.defaults.color = '#a0aec0';
Chart.defaults.borderColor = '#2d4a7c';
Chart.defaults.font.family = "'Inter', sans-serif";

const segmentColors = [
    '#667eea', // Premium Loyal
    '#f093fb', // Big Ticket
    '#43e97b', // Frequent
    '#fa709a', // Budget
    '#4facfe'  // At-Risk
];

const segmentLabels = ['Premium Loyal', 'Big Ticket', 'Frequent', 'Budget', 'At-Risk'];

// ============================================
// Initialize All Charts
// ============================================

function initCharts() {
    initClusterDistributionChart();
    initRevenueChart();
    initAgeSpendingChart();
    initIncomeSpendingChart();
    initPurchaseFreqChart();
    initRecencyChart();
    initModelComparisonChart();
    initKMeansChart();
}

// ============================================
// Cluster Distribution Chart
// ============================================

let clusterDistributionChart = null;

function initClusterDistributionChart() {
    const ctx = document.getElementById('clusterDistributionChart');
    if (!ctx) return;
    
    const data = {
        labels: segmentLabels,
        datasets: [{
            label: 'Customers',
            data: [1850, 1450, 2650, 2250, 1800],
            backgroundColor: segmentColors,
            borderColor: segmentColors.map(c => c),
            borderWidth: 0,
            borderRadius: 8,
            hoverOffset: 10
        }]
    };
    
    const config = {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: '#1a1a2e',
                    titleColor: '#ffffff',
                    bodyColor: '#a0aec0',
                    borderColor: '#2d4a7c',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${value.toLocaleString()} customers (${percentage}%)`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 11
                        }
                    }
                },
                y: {
                    grid: {
                        color: '#2d4a7c',
                        drawBorder: false
                    },
                    ticks: {
                        font: {
                            size: 11
                        }
                    }
                }
            }
        }
    };
    
    clusterDistributionChart = new Chart(ctx, config);
}

// ============================================
// Revenue Chart (Doughnut)
// ============================================

let revenueChart = null;

function initRevenueChart() {
    const ctx = document.getElementById('revenueChart');
    if (!ctx) return;
    
    const data = {
        labels: segmentLabels,
        datasets: [{
            data: [8.36, 5.58, 4.90, 2.07, 1.22],
            backgroundColor: segmentColors,
            borderColor: '#16213e',
            borderWidth: 3,
            hoverOffset: 15
        }]
    };
    
    const config = {
        type: 'doughnut',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 15,
                        usePointStyle: true,
                        pointStyle: 'circle',
                        font: {
                            size: 11
                        }
                    }
                },
                tooltip: {
                    backgroundColor: '#1a1a2e',
                    titleColor: '#ffffff',
                    bodyColor: '#a0aec0',
                    borderColor: '#2d4a7c',
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            return `$${value.toFixed(2)}M`;
                        }
                    }
                }
            }
        }
    };
    
    revenueChart = new Chart(ctx, config);
}

// ============================================
// Age vs Spending Score Chart
// ============================================

let ageSpendingChart = null;

function initAgeSpendingChart() {
    const ctx = document.getElementById('ageSpendingChart');
    if (!ctx) return;
    
    // Generate sample scatter data
    const generateClusterData = (baseX, baseY, spread, count) => {
        return Array.from({ length: count }, () => ({
            x: baseX + (Math.random() - 0.5) * spread,
            y: baseY + (Math.random() - 0.5) * spread
        }));
    };
    
    const datasets = [
        {
            label: 'Premium Loyal',
            data: generateClusterData(45, 75, 25, 50),
            backgroundColor: 'rgba(102, 126, 234, 0.6)',
            borderColor: '#667eea',
            pointRadius: 5,
            pointHoverRadius: 8
        },
        {
            label: 'Big Ticket',
            data: generateClusterData(52, 65, 20, 40),
            backgroundColor: 'rgba(240, 147, 251, 0.6)',
            borderColor: '#f093fb',
            pointRadius: 5,
            pointHoverRadius: 8
        },
        {
            label: 'Frequent',
            data: generateClusterData(35, 45, 30, 60),
            backgroundColor: 'rgba(67, 233, 123, 0.6)',
            borderColor: '#43e97b',
            pointRadius: 5,
            pointHoverRadius: 8
        },
        {
            label: 'Budget',
            data: generateClusterData(38, 25, 25, 45),
            backgroundColor: 'rgba(250, 112, 154, 0.6)',
            borderColor: '#fa709a',
            pointRadius: 5,
            pointHoverRadius: 8
        },
        {
            label: 'At-Risk',
            data: generateClusterData(48, 20, 25, 35),
            backgroundColor: 'rgba(79, 172, 254, 0.6)',
            borderColor: '#4facfe',
            pointRadius: 5,
            pointHoverRadius: 8
        }
    ];
    
    const config = {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        padding: 15,
                        usePointStyle: true,
                        pointStyle: 'circle',
                        font: {
                            size: 10
                        }
                    }
                },
                tooltip: {
                    backgroundColor: '#1a1a2e',
                    titleColor: '#ffffff',
                    bodyColor: '#a0aec0',
                    borderColor: '#2d4a7c',
                    borderWidth: 1,
                    padding: 12
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Age',
                        color: '#a0aec0',
                        font: {
                            size: 12,
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: '#2d4a7c'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Spending Score',
                        color: '#a0aec0',
                        font: {
                            size: 12,
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: '#2d4a7c'
                    },
                    min: 0,
                    max: 100
                }
            }
        }
    };
    
    ageSpendingChart = new Chart(ctx, config);
}

// ============================================
// Income vs Total Spending Chart
// ============================================

let incomeSpendingChart = null;

function initIncomeSpendingChart() {
    const ctx = document.getElementById('incomeSpendingChart');
    if (!ctx) return;
    
    const data = {
        labels: segmentLabels,
        datasets: [{
            label: 'Avg Income ($)',
            data: [125000, 95000, 55000, 35000, 42000],
            backgroundColor: 'rgba(102, 126, 234, 0.7)',
            borderColor: '#667eea',
            borderWidth: 2,
            borderRadius: 6,
            order: 2
        }, {
            label: 'Avg Spending ($)',
            data: [4520, 3850, 1850, 920, 680],
            backgroundColor: 'rgba(67, 233, 123, 0.7)',
            borderColor: '#43e97b',
            borderWidth: 2,
            borderRadius: 6,
            order: 1
        }]
    };
    
    const config = {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        padding: 15,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    grid: {
                        color: '#2d4a7c'
                    },
                    ticks: {
                        callback: function(value) {
                            if (value >= 1000) {
                                return '$' + (value / 1000).toFixed(0) + 'K';
                            }
                            return '$' + value;
                        }
                    }
                }
            }
        }
    };
    
    incomeSpendingChart = new Chart(ctx, config);
}

// ============================================
// Purchase Frequency Chart
// ============================================

let purchaseFreqChart = null;

function initPurchaseFreqChart() {
    const ctx = document.getElementById('purchaseFreqChart');
    if (!ctx) return;
    
    const data = {
        labels: ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '40+'],
        datasets: [{
            label: 'Customers',
            data: [450, 680, 920, 1250, 1580, 1890, 1450, 1020, 760],
            backgroundColor: (context) => {
                const chart = context.chart;
                const {ctx, chartArea} = chart;
                if (!chartArea) return null;
                const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
                gradient.addColorStop(0, 'rgba(102, 126, 234, 0.2)');
                gradient.addColorStop(1, 'rgba(102, 126, 234, 0.8)');
                return gradient;
            },
            borderColor: '#667eea',
            borderWidth: 2,
            borderRadius: 6,
            fill: true
        }]
    };
    
    const config = {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Number of Purchases',
                        color: '#a0aec0'
                    }
                },
                y: {
                    grid: {
                        color: '#2d4a7c'
                    },
                    title: {
                        display: true,
                        text: 'Customer Count',
                        color: '#a0aec0'
                    }
                }
            },
            elements: {
                line: {
                    tension: 0.4
                }
            }
        }
    };
    
    purchaseFreqChart = new Chart(ctx, config);
}

// ============================================
// Recency Chart
// ============================================

let recencyChart = null;

function initRecencyChart() {
    const ctx = document.getElementById('recencyChart');
    if (!ctx) return;
    
    const data = {
        labels: ['0-30', '31-60', '61-90', '91-120', '121-150', '151-180', '180+'],
        datasets: [{
            label: 'Customers',
            data: [2850, 2450, 1680, 1120, 890, 580, 430],
            backgroundColor: segmentColors.map(color => color + '99'),
            borderColor: segmentColors,
            borderWidth: 2,
            borderRadius: 6
        }]
    };
    
    const config = {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Days Since Last Purchase',
                        color: '#a0aec0'
                    }
                },
                y: {
                    grid: {
                        color: '#2d4a7c'
                    },
                    title: {
                        display: true,
                        text: 'Customer Count',
                        color: '#a0aec0'
                    }
                }
            }
        }
    };
    
    recencyChart = new Chart(ctx, config);
}

// ============================================
// Model Comparison Chart
// ============================================

let modelComparisonChart = null;

function initModelComparisonChart() {
    const ctx = document.getElementById('modelComparisonChart');
    if (!ctx) return;
    
    const data = {
        labels: ['K-Means', 'Hierarchical', 'DBSCAN', 'GMM'],
        datasets: [{
            label: 'Silhouette Score',
            data: [0.32, 0.28, 0.21, 0.26],
            backgroundColor: [
                'rgba(67, 233, 123, 0.8)',
                'rgba(102, 126, 234, 0.8)',
                'rgba(250, 112, 154, 0.8)',
                'rgba(79, 172, 254, 0.8)'
            ],
            borderColor: [
                '#43e97b',
                '#667eea',
                '#fa709a',
                '#4facfe'
            ],
            borderWidth: 2,
            borderRadius: 8
        }]
    };
    
    const config = {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    grid: {
                        color: '#2d4a7c'
                    },
                    max: 0.5,
                    title: {
                        display: true,
                        text: 'Silhouette Score',
                        color: '#a0aec0'
                    }
                },
                y: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    };
    
    modelComparisonChart = new Chart(ctx, config);
}

// ============================================
// K-Means Chart (PCA Visualization Simulation)
// ============================================

let kmeansChart = null;

function initKMeansChart() {
    const ctx = document.getElementById('kmeansChart');
    if (!ctx) return;
    
    // Simulated PCA scatter data
    const generatePCAData = (centerX, centerY, spread, count) => {
        return Array.from({ length: count }, () => ({
            x: centerX + (Math.random() - 0.5) * spread,
            y: centerY + (Math.random() - 0.5) * spread
        }));
    };
    
    const datasets = segmentColors.map((color, index) => ({
        label: `Cluster ${index}`,
        data: generatePCAData(
            Math.random() * 100,
            Math.random() * 100,
            30,
            30 + Math.floor(Math.random() * 20)
        ),
        backgroundColor: color + 'CC',
        borderColor: color,
        pointRadius: 4,
        pointHoverRadius: 6
    }));
    
    const config = {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    display: false,
                    min: 0,
                    max: 100
                },
                y: {
                    display: false,
                    min: 0,
                    max: 100
                }
            }
        }
    };
    
    kmeansChart = new Chart(ctx, config);
}

// ============================================
// Utility Functions
// ============================================

function destroyChart(chartInstance) {
    if (chartInstance) {
        chartInstance.destroy();
    }
}

function updateChartData(chartId, newData) {
    const chart = Chart.getChart(chartId);
    if (chart) {
        chart.data.datasets[0].data = newData;
        chart.update();
    }
}

