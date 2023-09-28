[ -f style.css ] && rm style.css && touch style.css

for file in $(ls css/); do
  yuicompressor --type css -o tmp.css ./css/$file
  cat tmp.css >> style.css
  rm tmp.css
done
