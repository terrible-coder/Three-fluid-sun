const ghpages = require("gh-pages");

ghpages.publish(".", {
	silent: true,
	dotfiles: true,
	add: true,
	user: {
		name: "Ayanava De",
		email: "ayanavade01@gmail.com"
	},
	repo: `https://${process.env.GH_TOKEN}@github.com/terrible-coder/Three-fluid-sun`,
	message: "Saving to gh-pages"
}, err => {
	if(err !== undefined) {
		console.log(err);
		throw new Error("Something went wrong.");
	} else console.log("Published.");
})