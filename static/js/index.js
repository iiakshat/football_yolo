function showform() {
    var checkBox = document.getElementById("advanced");
    var config = document.getElementById("configform");
    if (checkBox.checked == true){
        config.style.display = "block";
    } else {
        config.style.display = "none";
    }
}

function updateAlpha(value) {
    document.getElementById('alphaOutput').value = value;
}