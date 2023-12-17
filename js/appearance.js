import { app } from "../../scripts/app.js";

const COLOR_THEMES = {
    red: { nodeColor: "#332222", nodeBgColor: "#553333" },
    green: { nodeColor: "#223322", nodeBgColor: "#335533" },
    blue: { nodeColor: "#222233", nodeBgColor: "#333355" },
    pale_blue: { nodeColor: "#2a363b", nodeBgColor: "#3f5159" },
    cyan: { nodeColor: "#223333", nodeBgColor: "#335555" },
    purple: { nodeColor: "#332233", nodeBgColor: "#553355" },
    yellow: { nodeColor: "#443322", nodeBgColor: "#665533" },
    none: { nodeColor: null, nodeBgColor: null } // no color
};

const NODE_COLORS = {
    "KSampler (Efficient)": "random",
    "KSampler Adv. (Efficient)": "random",
    "KSampler SDXL (Eff.)": "random",
    "Efficient Loader": "random",
    "Eff. Loader SDXL": "random",
    "LoRA Stacker": "blue",
    "Checkpoint Config": "blue",
    "Control Net Stacker": "green",
    "Apply ControlNet Stack": "none",
    "XY Plot": "purple",
    "Unpack SDXL Tuple": "none",
    "Pack SDXL Tuple": "none",
    "XY Input: Seeds++ Batch": "cyan",
    "XY Input: Add/Return Noise": "cyan",
    "XY Input: Steps": "cyan",
    "XY Input: CFG Scale": "cyan",
    "XY Input: Sampler/Scheduler": "cyan",
    "XY Input: Denoise": "cyan",
    "XY Input: VAE": "cyan",
    "XY Input: Prompt S/R": "cyan",
    "XY Input: Aesthetic Score": "cyan",
    "XY Input: Refiner On/Off": "cyan",
    "XY Input: Checkpoint": "cyan",
    "XY Input: Clip Skip": "cyan",
    "XY Input: LoRA": "cyan",
    "XY Input: LoRA Plot": "cyan",
    "XY Input: LoRA Stacks": "cyan",
    "XY Input: Control Net": "cyan",
    "XY Input: Control Net Plot": "cyan",
    "XY Input: Manual XY Entry": "cyan",
    "Manual XY Entry Info": "cyan",
    "Join XY Inputs of Same Type": "cyan",
    "Image Overlay": "random",
    "Noise Control Script": "none",
    "HighRes-Fix Script": "yellow",
    "Tiled Upscaler Script": "red",
    "AnimateDiff Script": "random",
    "Evaluate Integers": "pale_blue",
    "Evaluate Floats": "pale_blue",
    "Evaluate Strings": "pale_blue",
    "Simple Eval Examples": "pale_blue",
 };

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];  // Swap elements
    }
}

let colorKeys = Object.keys(COLOR_THEMES).filter(key => key !== "none");
shuffleArray(colorKeys);  // Shuffle the color themes initially

function setNodeColors(node, theme) {
    if (!theme) {return;}
    node.shape = "box";
    if(theme.nodeColor && theme.nodeBgColor) {
        node.color = theme.nodeColor;
        node.bgcolor = theme.nodeBgColor;
    }
}

const ext = {
    name: "efficiency.appearance",

    nodeCreated(node) {
        const nclass = node.comfyClass;
        if (NODE_COLORS.hasOwnProperty(nclass)) {
            let colorKey = NODE_COLORS[nclass];

            if (colorKey === "random") {
                // Check for a valid color key before popping
                if (colorKeys.length === 0 || !COLOR_THEMES[colorKeys[colorKeys.length - 1]]) {
                    colorKeys = Object.keys(COLOR_THEMES).filter(key => key !== "none");
                    shuffleArray(colorKeys);
                }
                colorKey = colorKeys.pop();
            }

            const theme = COLOR_THEMES[colorKey];
            setNodeColors(node, theme);
        }
    }
};

app.registerExtension(ext);
