<?php
if($_POST["Message"]) {
mail("judy9906@gmail.com", "Contact from Judy Park",
$_POST["Insert Your Message"]. "From: Judie1999@github.io");
}
?>