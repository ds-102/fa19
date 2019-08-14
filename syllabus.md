---
layout: page
title: "Syllabus"
page_class: content-wide
---

This syllabus is still under development and is subject to change.

<script>
  function get_buttom() {
    return document.getElementsByClassName('show_hide_description_click')[0];
  }

  function showCurrentWeekDescription() {
    const lectures = document.getElementsByClassName('lecture');

  for (var i = 0; i < lectures.length; i++ ) {
      let lecture = lectures[i];
      const { lectureWeek, lectureDate } = lecture.dataset;
      const lec_date = new Date(lectureDate + ' 23:59:59');

      if (current_date <= lec_date) {
        const descEls = document.getElementsByClassName(`description-week-${lectureWeek}`);
        for (var j = 0; j < descEls.length; j++) {
          descEl = descEls[j];
          descEl.hidden = null;
        }
        break;
      }
    }
  }

  function hideAllDescription() {
    const descEls = document.getElementsByClassName('lecture__description');
    for (var j = 0; j < descEls.length; j++) {
      descEl = descEls[j];
      descEl.hidden = "true";
    }
    showCurrentWeekDescription();

    const buttom = get_buttom();
    buttom.text = "Show all lecture descriptions";
    buttom.onclick=showAllDescription;
  }

  function showAllDescription() {
    const descEls = document.getElementsByClassName('lecture__description');
    for (var j = 0; j < descEls.length; j++) {
      descEl = descEls[j];
      descEl.hidden = null;
    }

    const buttom = get_buttom();
    buttom.text = "Hide all lecture descriptions";
    buttom.onclick=hideAllDescription;
  }

</script>
<a class="show_hide_description_click" href="javascript:void(0)" onclick="showAllDescription();">Show all lecture descriptions</a>


<div class="table-responsive">
  <table class="syllabus table" id="syllabus_table">
    <colgroup>
      <col width="65px">
      <col width="78px">
      <col width="115px">
      <col width="">
    </colgroup>
    <thead>
      <tr class="syllabus__header">
        <th> Week </th>
        <th> Lecture </th>
        <th> Date </th>
        <th> Topic </th>
        <th> Lab </th>
        <th> Discussion </th>
        <th> Homework </th>
      </tr>
    </thead>
    <tbody>

    <!--
    The actual lecture rows. To add a lecture, edit _data/lectures.yml.
    -->

    {% include syllabus_entries.html %}


    </tbody>
  </table>
</div>

<!--
Script to highlight the current lecture.
-->

<script type="text/javascript">
const current_date = new Date();
const lectures = document.getElementsByClassName('lecture');

for (var i = 0; i < lectures.length; i++ ) {
  let lecture = lectures[i];
  const { lectureWeek, lectureDate } = lecture.dataset;
  const lec_date = new Date(lectureDate + ' 23:59:59');

  // We need to find the first occurance of lecture that pass today's date
  if (current_date <= lec_date) {
    lecture.className += ' lecture--current';

    // Need to look up the week element since it might be in the row above
    const weekEl = document.getElementById(`lecture-week-${lectureWeek}`);
    weekEl.className += ' lecture__week--current';

    // We will show the description for lectures in the coming week
    const descEls = document.getElementsByClassName(`description-week-${lectureWeek}`);
    for (var j = 0; j < descEls.length; j++) {
      descEl = descEls[j];
      descEl.hidden = null;
    }

    break;
  }

  window.location.hash = `lecture-week-${lectureWeek}`;
}
</script>

