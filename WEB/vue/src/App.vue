<template>
<v-app>
  <div class="bg" v-bind:class="{'blur-content':dialog}">
    <div class="container">
      <div class="row">
        <div class="col-md-6 mx-auto">
          <h1 class="text-center">YouTube Hate Video Detector</h1>
          <form @submit="image_process">
            <label for="url"></label>
            <input type="text" v-model="url" id="url" class="form-control" />
            <button type="submit" class="btn btn-success btn-block mt-3">Submit</button>
          </form>
        </div>
      </div>
    </div>
    <div class="container pt-3" v-if="thumbnailFlag">
      <div class="row">
        <div class="col-md-5 mx-auto">
          <div class="card">
            <img v-bind:src="thumbnail" class="card-img-top" alt="..." />
            <div class="card-body" v-if="SentimentFlag">
              <h6 class="card-title"><b>{{thumbnailText}}</b></h6>
              <h6 class="card-text"><b>
                Sentiment : {{sentiment}}</b></h6>
                <h6><b>Category : {{category}}</b></h6>
              <div class="progress" style="height: 20px;">
                <div
                  v-if="hatePercentage >= 50"
                  class="progress-bar progress-bar-striped bg-danger"
                  role="progressbar"
                  v-bind:style="{width: hatePercentage + '%'}"
                  v-bind:aria-valuenow="hatePercentage"
                  aria-valuemin="0"
                  aria-valuemax="100"
                ><b>Hate Level : {{hatePercentage + '%'}}</b></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
     <loader v-bind:dialog="dialog"></loader> 
  </div>
  </v-app>
</template>

<script>
import loader from './components/loader'
import axios from "axios";
export default {
  components:{loader},
  name: "app",
  data() {
    return {
      dialog:false,
      url: "",
      thumbnailFlag: false,
      thumbnail: "",
      SentimentFlag: false,
      sentiment: "",
      hateLevel: 0.0,
      category: "",
      thumbnailText: "",
      hatePercentage: 0.0,
      vid: ""
    };
  },
  methods: {
    image_process(e) {
      e.preventDefault();
      if (this.url != "") {
        this.dialog=true;
        this.vid = this.url.substring(32);
        console.log(this.vid);
        this.thumbnail =
          "https://img.youtube.com/vi/" + this.vid + "/hqdefault.jpg";
        this.thumbnailFlag = true;
        axios
          .post("http://localhost:5000/image_process", {
            url: this.url
          })
          .then(res => {
          this.thumbnailText = res.data.thumbnail_text;
            // this.pre_process();
            console.log(res.data);
          this.sentiment = res.data.sentiment;
          this.hateLevel = parseFloat(res.data.hateLevel);
          this.hatePercentage = this.hateLevel * 100;
          this.category = res.data.category;
          this.SentimentFlag = true;
          this.dialog=false;
          })
          .catch(err => console.log(err));
      }
    } //,
    // pre_process() {
    //   console.log(this.thumbnailText);
    //   axios
    //     .post("http://localhost:5000/pre_process", {
    //       url: this.url,
    //       thumbnail_text: this.thumbnailText
    //     })
    //     .then(res => {
    //       // this.thumbnailText = res.data.thumbnail_text;
    //       // this.text_process();
    //       console.log(res.data);
    //       this.sentiment = res.data.sentiment;
    //       this.hateLevel = parseFloat(res.data.hateLevel);
    //       this.hatePercentage = this.hateLevel * 100;
    //       this.category = res.data.category;
    //       this.SentimentFlag = true;
    //     })
    //     .catch(err => console.log(err));
    // } ,
    // text_process() {
    //   axios
    //     .get("http://localhost:5000/text_process")
    //     .then(res => {
    //       console.log(res.data);
    //       this.sentiment = res.data.sentiment;
    //       this.hateLevel = parseFloat(res.data.hateLevel);
    //       this.hatePercentage = this.hateLevel * 100;
    //       this.category = res.data.category;
    //       this.SentimentFlag = true;
    //     })
    //     .catch(err => console.log(err));
    // }
  }
};
</script>

<style>
.bg {
  /* background-image: url("./assets/bg.jpg"); */
  background: url(./assets/bg.jpg) no-repeat center center fixed;
  background-size: cover;
  height: 100%;
  overflow: hidden;  
  height: 100vh;
  width: 100vw;
  font-size: 20px;
  margin: 0px;
}
/* @import'~bootstrap/dist/css/bootstrap.css' */
.blur-content{
  filter: blur(5px); 
}
</style>