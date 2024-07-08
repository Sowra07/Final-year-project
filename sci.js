function showLinearRegression() {
    fetch('/linear_regression')  // Replace with your backend endpoint
        .then(response => response.text())
        .then(data => {
            document.getElementById('result-container').innerHTML = data;
            document.getElementById('result-container').style.display = 'block';
        });
}

function showRandomForest() {
    fetch('/random_forest')  // Replace with your backend endpoint
        .then(response => response.text())
        .then(data => {
            document.getElementById('result-container').innerHTML = data;
            document.getElementById('result-container').style.display = 'block';
        });
}

function showNeuralNetwork() {
    fetch('/neural_network')  // Replace with your backend endpoint
        .then(response => response.text())
        .then(data => {
            document.getElementById('result-container').innerHTML = data;
            document.getElementById('result-container').style.display = 'block';
        });
}

function showComparison() {
    fetch('/comparison')  // Replace with your backend endpoint
        .then(response => response.text())
        .then(data => {
            document.getElementById('result-container').innerHTML = data;
            document.getElementById('result-container').style.display = 'block';
        });
    
}

