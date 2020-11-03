function bmi() {
    let height = parseInt(document.getElementById("height").value) / 100
    let weight = parseInt(document.getElementById("weight").value)

    let bmi = Math.round(((weight / (Math.pow(height, 2))) * 10)) / 10
    console.log(bmi)
}