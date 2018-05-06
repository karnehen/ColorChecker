# ColorChecker

This is a tool for calibrating and describing the colors of weed seeds on a photo.
The main goal is to complement the SeedCounter app (https://play.google.com/store/apps/details?id=org.wheatdb.seedcounter)
with color recognition capabilities.
The tool was written as a project in Computer Science Center (https://compscicenter.ru/)

## Workflow
* Input image should be a photo with seeds on a white sheet of paper and X-rite color checker.
* The color checker is searched with a help of a reference image.
It probably can be changed to adapt different models of color checker but this was not tested.
* After that the found color checker is used as a reference for color correction.
* The result is the same image with corrected colors (and optionally with background-filled color checker).
* For weed seeds the features characterizing the seed kind can be retrieved (not implemented yet :)).

## Examples
TODO: provide a description of examples

## Known issues
* The accuracy of color checker retrieval is not perfect.
* The calibrated image can sometimes look unnatural (especially with high dimension regression models).
* The performance on limited resources (like an Android phone) was not tested.
