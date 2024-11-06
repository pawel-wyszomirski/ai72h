window.addEventListener("chainlit-call-fn", (e) => {
    const { name, args, callback } = e.detail;
    callback("You sent: " + args.msg);
});

window.mountChainlitWidget({
    chainlitServer: "https://YOUR_APP_NAME.ploomberapp.io",
});