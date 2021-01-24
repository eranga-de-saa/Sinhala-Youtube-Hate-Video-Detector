<template>
  <div class="text-center">
    <!-- <v-btn
      :disabled="dialog"
      :loading="dialog"
      class="white--text"
      color="purple darken-2"
      @click="dialog = true"
    ></v-btn> -->


    <v-dialog v-model="dialog" hide-overlay persistent width="300">
      <v-card color="danger" dark >
        <v-card-text>
         <b> Processing .......</b>
          <v-progress-linear indeterminate color="white" class="mb-0"></v-progress-linear>
        </v-card-text>
      </v-card>
    </v-dialog>
  </div>
</template>

<script>
export default {
    props:['dialog']
}
</script>
