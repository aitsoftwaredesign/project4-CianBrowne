function displayProgram(program) {
    console.log(program)
    switch (program) {
        case 0:
            var x = document.getElementById("programFittest");
            x.style.display = "block";
            break;
        case 1:
            var x = document.getElementById("program2ndFittest");
            x.style.display = "block";
            break;
        case 2:
            var x = document.getElementById("program3rdFittest");
            x.style.display = "block";
            break;
        case 3:
            var x = document.getElementById("program4thFittest");
            x.style.display = "block";
            break;
        case 4:
            var x = document.getElementById("program5thFittest");
            x.style.display = "block";
            break;
    }

}