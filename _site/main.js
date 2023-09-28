//credit: James Hannah @cyborgsphinx
function clearContent(kind) {
	var i;
	var contentClass = kind + "-content";
	var tablinkClass = kind + "-tablinks";

	var content = document.getElementsByClassName(contentClass);
	for (i = 0; i < content.length; i++) {
		content[i].style.display = "none";
	}

	var tablinks = document.getElementsByClassName(tablinkClass);
	for (i = 0; i < tablinks.length; i++) {
		tablinks[i].className = tablinks[i].className.replace(" active", "");
	}
}

function changeStation(event) {
	var i;
	var target = event.currentTarget;
	var station = target.id;

	clearContent("station")  // affects station-content and station-tablinks

	var showContent = document.getElementsByClassName(station);
	for (i = 0; i < showContent.length; i++) {
		showContent[i].style.display = "block";
	}
	target.className += " active";

	window.sessionStorage.setItem("station", station)
}

function changeSubtab(event) {
	var i;
	var target = event.currentTarget;
	var subtab = target.id;

	clearContent("subtab")  // affects subtab-content and subtab-tablinks

	document.getElementById(subtab + "-content").style.display = "block";
	target.className += " active";

	window.sessionStorage.setItem("subtab", subtab)
}

if (window.sessionStorage.getItem("station")) {
	document.getElementById(window.sessionStorage.getItem("station")).click();
} else {
	// choose default
	document.getElementById("e01").click();
}
if (window.sessionStorage.getItem("subtab")) {
	document.getElementById(window.sessionStorage.getItem("subtab")).click();
} else {
	// choose default
	document.getElementById("raw-data").click();
