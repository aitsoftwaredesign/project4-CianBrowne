$(document).ready(function(){
    $("#registration").hide();
})

function showReg(){
    $("#registration").show();
    $("#login").hide();
}
function showLogin(){
    $("#registration").hide();
    $("#login").show();
}