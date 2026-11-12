# Interactive Views of the CLIP Retrieval Space

## Page layout

One page, three views selectable by clicking the labels at the top: semantic map, Grad-CAM and FovEx. The sidebar fixes the image, the target caption and the grid size. The two models, zero-shot baseline and fine-tuned, sit side by side and share the query.

The explanation views are fully controllable. HTML, CSS, and JS are all independently implemented, so layout, colors, events, and hover logic can be modified directly without relying on third-party libraries.

The target caption is one of ten candidates. These consist of the five reference captions of the image plus one caption from each of the five nearest images. The first five are positives, the last five are hard negatives.

The semantic map is a Streamlit and Plotly chart, so zoom, pan and hover come from Plotly itself. Grad-CAM and FovEx are a small web page written for this project. One document holds both models, and a single hover on a grid cell swaps the image, the small image and the text in both panels at once. FovEx also separates three states, an ordinary cell, an intermediate fixation and the final fixation.

## Semantic map

Grey points are captions, coloured points are images, and the colour is the supercategory of the dominant object; `modality` turns all images blue. The five reference captions of the selected image are red stars, the eight captions with the highest cosine to the image are orange diamonds, and the image itself is a ring, black on the light theme and white on the dark theme. The legend holds the caption markers, the selected image and the supercategories.

Hover shows the caption text or the image id; the wheel zooms and the drag pans. In `compare` the two models sit side by side, but each layout is fitted on its own, so positions do not compare across panels. Read the structure instead, do the red stars fall inside the image cloud, do the orange diamonds sit around the selected image.

2D distance is not similarity. t-SNE keeps local neighbourhoods, so a caption can be far from its image and still have a high cosine; the number in the red star hover is the value to read. Images and captions also tend to form two clouds, the modality gap, so a large distance does not mean the pair is unaligned.

## Grad-CAM

Zero-shot on the left, fine-tuned on the right, with a jet heatmap over the input. The default line gives the target score, the target probability over the ten candidates and the top caption.

The two panels are in the same document, so one hover updates both at the same time; the images and the four values are already prepared when the page is built.

When hovering over a cell:

- `CAM share`: the fraction of the total map mass in the cell.
- `CAM mean`: the average activation of the cell.
- `target score drop`: the loss of cosine to the target caption after the cell is replaced with grey, so a positive value means the cell supports the caption.
- `top caption after removal`: the winning candidate once the cell is hidden.

The small image under the panel is an upscaled crop of the cell.

Warm colour, large share and positive drop together mean the model relies on the cell. A warm cell with a drop near zero or negative is a large gradient that does not matter. The map is 7x7, about 32 pixels per cell, so a small object is diluted by its background; a cold cell is not proof that the model ignores the object.

## FovEx

The default state shows the input with the ten-fixation scanpath and the final attribution map in the small image. Hover has three cases:

- A cell without a fixation simulates a single gaze: sharp inside that cell, blurred elsewhere. The small image is the Grad-CAM of that view, and the text gives the probability and the top caption at that gaze.
- An intermediate fixation shows the accumulated view at that step and the path walked so far, with the accumulated mask in the small image.
- The final fixation shows the complete FovEx map.

When several fixations share a cell, the last one is shown and the final fixation wins. With a 1x1 grid the whole view is a static picture with the final result. The three cases come from the same hover rule, which picks the view, the small image and the text for the cell under the pointer.

When hovering over a cell:

- `target probability`: the softmax over the ten candidates, read as a relative value within one model. Each hover state also gives the difference to the full-image probability, so the sign shows whether the gaze helps or hurts the caption. The probability stays high while the fixation is on the object named in the caption and falls on the background. The two models usually differ in scanpath and attribution, and the difference comes from the representation they learned.
- `target caption`: the candidate selected in the sidebar; the fixation sequence is optimised for it, so a different target gives a different scanpath and different states below.
- `top caption`: the highest-scoring candidate under the current view; it can differ from the target.
- `simulated fixation`: a cell without a fixation point; only a Gaussian window at the cell is sharp and the rest is blurred, and the top caption comes from that single view.
- `fixation N` / `step N of 10`: a cell that contains the N-th fixation; the view accumulates the first N foveae, and the top caption comes from that accumulated view.
- `final fixation`: the cell of the last fixation; the complete FovEx map is shown.
- The same cell can be a simulated fixation under one target and a fixation under another; the top caption follows the view, and the probability difference is relative to the full image of the same target.

If you want to select a cell in the middle and then move the mouse elsewhere without changing the selected cell, simply right-click the selected cell and then move the mouse out of the image area.

## Grid size

The slider runs from 1 to 6 with a default of 4. A finer grid gives smaller cells, but the compute and the page size grow with n squared and the hover targets get small. An infinite grid would make the simulated gaze continuous, but the FovEx sequence would still be ten fixations; the CAM is 7x7 anyway, so cells below about 32 pixels add no map detail. 4 to 6 is enough.
