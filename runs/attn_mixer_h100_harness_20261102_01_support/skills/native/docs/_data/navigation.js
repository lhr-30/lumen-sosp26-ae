import tutorials from "./tutorials.js";
import languageReference from "./languageReference.js";

export default [
  {
    title: "Get Started",
    items: [
      { title: "Installation", url: "get-started/installation/" }
    ]
  },
  {
    title: "Tutorials",
    items: [
      { title: "Overview", url: "tutorials/" },
      ...tutorials
    ]
  },
  {
    title: "Language Reference",
    items: [
      { title: "Overview", url: "language-reference/" },
      ...languageReference
    ]
  },
  {
    title: "Development Tips",
    items: [
      { title: "Overview", url: "development-tips/" },
      { title: "Debugging", url: "development-tips/debugging/" }
    ]
  }
];
