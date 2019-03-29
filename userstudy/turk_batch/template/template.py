
TEMPL = """
<style>
th, td {{
    border: 2px solid black;
    padding: 8px;
}}

th {{
    width: 100px;
    vertical-align: text-top;
    font-weight: normal;
}}

.noborder {{
    border: 0px solid black;
}}

.thlab {{
    margin-bottom: 1em;
}}

.td {{
    text-align: center;
    vertical-align: middle;
}}

input[type=radio] {{
    border: 0px;
    width: 100%;
    height: 4em;
}}

audio {{
    width: 300px;
}}

input[type=submit] {{
    margin-top: 20px;
    width: 20em;
    height: 2em;
}}

</style>

<html>

<h2>How natural are these speech recordings?</h2>

<p><b>Please use headphones in a quiet environment if possible.</b></p>
<p>You will be presented a batch of recordings and asked to rate each of their naturalness.</p>
<p>These recordings will all be of the same sentence but differ in subtle ways.</p>
<p><b>Feel free to listen to each recording as many times as you like and update your scores as you compare the methods.</b></p>

<div class="form-group">

<table>
	<tbody>
		<tr>
			<th class="noborder"></th>
			<th><div class="thlab"><b>1: Bad</b></div><div>Completely unnatural speech</div></th>
			<th><div class="thlab"><b>2: Poor</b></div><div>Mostly unnatural speech</div></th>
			<th><div class="thlab"><b>3: Fair</b></div><div>Equally natural and unnatural speech</div></th>
			<th><div class="thlab"><b>4: Good</b></div><div>Mostly natural speech</div></th>
			<th><div class="thlab"><b>5: Excellent</b></div><div>Completely natural speech</div></th>
		</tr>
                {rows}
	</tbody>
</table>

<input type="submit">

</div>

</html>
"""

ROW_TEMPL = """
		<tr>
			<td><audio controls=""><source src="${{recording_{i}_url}}" type="audio/mpeg"/></audio></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_rating" value="1"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_rating" value="2"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_rating" value="3"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_rating" value="4"></td>
			<td><input class="form-control" type="radio" required="" name="recording_{i}_rating" value="5"></td>
		</tr>
"""

import sys

n = int(sys.argv[1])

rows = []
for i in range(n):
  rows.append(ROW_TEMPL.format(i=i))
rows = '\n'.join(rows)

print(TEMPL.format(rows=rows))
