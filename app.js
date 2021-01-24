$( document ).ready(function() {
    console.log( "ready!" );

$(".thumbnailFlag").hide();

$( "#urlButton" ).click(function(e) {
	 e.preventDefault();
  url = $('#url').val()
if(url != '')

vid = url.substring(32);
 // console.log(this.vid);
 thumbnail ="https://img.youtube.com/vi/" + vid + "/hqdefault.jpg";

$('img').attr('src', thumbnail);

$(".thumbnailFlag").show();
$(".SentimentFlag").hide();

postUrl = 'http://localhost:5000/image_process'
data = {url:vid}
//jdata = JSON.stringify(data)
 $.ajax({
  type: "GET",
  // dataType: "json",
  url: postUrl,
  data: data,
  success: function (data) {
        category = data.category;
        sentiment=data.sentiment;
        text=data.thumbnail_text;
        level=parseFloat(data.hateLevel);

$("#Sentiment").html("Sentiment : " + sentiment)
$("#Category").html("Category : " +category)
$("#Text").html(text)

hatePercentage = level*100;
// $(".progresscls").attr('style>width',hatePercentage+"%")
// $(".progresscls").attr('aria-valuenow',hatePercentage)
$('.progress-bar').css('width', hatePercentage+'%').attr('aria-valuenow', hatePercentage).html( "<b>Hate Level : " + hatePercentage+'% </b>');



        $(".SentimentFlag").show();
}
});



});


});
