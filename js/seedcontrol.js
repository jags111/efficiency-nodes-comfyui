import { app } from "../../scripts/app.js";
import { addMenuHandler } from "./node_options/common/utils.js";

const LAST_SEED_BUTTON_LABEL = '🎲 Randomize / ♻️ Last Queued Seed';
const SEED_BEHAVIOR_RANDOMIZE = 'Randomize';
const SEED_BEHAVIOR_INCREMENT = 'Increment';
const SEED_BEHAVIOR_DECREMENT = 'Decrement';

const NODE_WIDGET_MAP = {
    "KSampler (Efficient)": "seed",
    "KSampler Adv. (Efficient)": "noise_seed",
    "KSampler SDXL (Eff.)": "noise_seed",
    "Noise Control Script": "seed",
    "HighRes-Fix Script": "seed",
    "Tiled Upscaler Script": "seed"
};

const SPECIFIC_WIDTH = 325; // Set to desired width

function setNodeWidthForMappedTitles(node) {
     if (NODE_WIDGET_MAP[node.comfyClass]) {
        node.setSize([SPECIFIC_WIDTH, node.size[1]]);
    }
}

class SeedControl {
    constructor(node, seedName) {
        this.lastSeed = -1;
        this.serializedCtx = {};
        this.node = node;
        this.seedBehavior = 'fixed'; // Default behavior

        let controlAfterGenerateIndex;

        for (const [i, w] of this.node.widgets.entries()) {
            if (w.name === seedName) {
                this.seedWidget = w;
            } else if (w.name === 'control_after_generate') {
                controlAfterGenerateIndex = i;
                this.node.widgets.splice(i, 1);
            }
        }

        if (!this.seedWidget) {
            throw new Error('Something\'s wrong; expected seed widget');
        }

        this.lastSeedButton = this.node.addWidget("button", LAST_SEED_BUTTON_LABEL, null, () => {
            const isValidValue = Number.isInteger(this.seedWidget.value) && this.seedWidget.value >= min && this.seedWidget.value <= max;
            
            // Special case: if the current label is the default and seed value is -1
            if (this.lastSeedButton.name === LAST_SEED_BUTTON_LABEL && this.seedWidget.value == -1) {
                return; // Do nothing and return early
            }
            
            if (isValidValue && this.seedWidget.value != -1) {
                this.lastSeed = this.seedWidget.value;
                this.seedWidget.value = -1;
            } else if (this.lastSeed !== -1) {
                this.seedWidget.value = this.lastSeed;
            } else {
                this.seedWidget.value = -1; // Set to -1 if the label didn't update due to a seed value issue
            }
            
            if (isValidValue) {
                this.updateButtonLabel(); // Update the button label to reflect the change
            }
        }, { width: 50, serialize: false });

        setNodeWidthForMappedTitles(node);
        if (controlAfterGenerateIndex !== undefined) {
            const addedWidget = this.node.widgets.pop();
            this.node.widgets.splice(controlAfterGenerateIndex, 0, addedWidget);
            setNodeWidthForMappedTitles(node);
        }

        const max = Math.min(1125899906842624, this.seedWidget.options.max);
        const min = Math.max(-1125899906842624, this.seedWidget.options.min);
        const range = (max - min) / (this.seedWidget.options.step / 10);

        this.seedWidget.serializeValue = async (node, index) => {
            // Check if the button is disabled
            if (this.lastSeedButton.disabled) {
                return this.seedWidget.value;
            }

            const currentSeed = this.seedWidget.value;
            this.serializedCtx = {
                wasSpecial: currentSeed == -1,
            };

            if (this.serializedCtx.wasSpecial) {
                switch (this.seedBehavior) {
                    case 'increment':
                        this.serializedCtx.seedUsed = this.lastSeed + 1;
                        break;
                    case 'decrement':
                        this.serializedCtx.seedUsed = this.lastSeed - 1;
                        break;
                    default:
                        this.serializedCtx.seedUsed = Math.floor(Math.random() * range) * (this.seedWidget.options.step / 10) + min;
                        break;
                }

            // Ensure the seed value is an integer and remains within the accepted range
            this.serializedCtx.seedUsed = Number.isInteger(this.serializedCtx.seedUsed) ? Math.min(Math.max(this.serializedCtx.seedUsed, min), max) : this.seedWidget.value;

            } else {
                this.serializedCtx.seedUsed = this.seedWidget.value;
            }

            if (node && node.widgets_values) {
                node.widgets_values[index] = this.serializedCtx.seedUsed;
            } else {
                // Update the last seed value and the button's label to show the current seed value
                this.lastSeed = this.serializedCtx.seedUsed;
                this.updateButtonLabel();
            }

            // Don't update the widget value to maintain -1 in the UI when seed was special
            if (!this.serializedCtx.wasSpecial) {
                this.seedWidget.value = this.serializedCtx.seedUsed;
            }

            if (this.serializedCtx.wasSpecial) {
                this.lastSeed = this.serializedCtx.seedUsed;
                this.updateButtonLabel();
            }

            return this.serializedCtx.seedUsed;
        };

        this.seedWidget.afterQueued = () => {
            // Check if the button is disabled
            if (this.lastSeedButton.disabled) {
                return; // Exit the function immediately
            }

            // Update lastSeed if user manually changed the seed to a specific value
            if (!this.serializedCtx.wasSpecial && this.seedWidget.value !== -1) {
                this.lastSeed = this.seedWidget.value;
            }

            this.updateButtonLabel();
            this.serializedCtx = {};
        };
    }

    setBehavior(behavior) {
        this.seedBehavior = behavior;

        // Capture the current seed value as lastSeed and then set the seed widget value to -1
        if (this.seedWidget.value != -1) {
            this.lastSeed = this.seedWidget.value;
            this.seedWidget.value = -1;
        }

        this.updateButtonLabel();
    }

    updateButtonLabel() {

        switch (this.seedBehavior) {
            case 'increment':
                this.lastSeedButton.name = `➕ Increment / ♻️ ${this.lastSeed === -1 ? "Last Queued Seed" : this.lastSeed}`;
                break;
            case 'decrement':
                this.lastSeedButton.name = `➖ Decrement / ♻️ ${this.lastSeed === -1 ? "Last Queued Seed" : this.lastSeed}`;
                break;
            default:
                this.lastSeedButton.name = `🎲 Randomize / ♻️ ${this.lastSeed === -1 ? "Last Queued Seed" : this.lastSeed}`;
                break;
        }
    }

}

function showSeedBehaviorMenu(value, options, e, menu, node) {
    const behaviorOptions = [
        {
            content: "🎲 Randomize",
            callback: () => {
                node.seedControl.setBehavior('randomize');
            }
        },
        {
            content: "➕ Increment",
            callback: () => {
                node.seedControl.setBehavior('increment');
            }
        },
        {
            content: "➖ Decrement",
            callback: () => {
                node.seedControl.setBehavior('decrement');
            }
        }
    ];

    new LiteGraph.ContextMenu(behaviorOptions, {
        event: e,
        callback: null,
        parentMenu: menu,
        node: node
    });

    return false;  // This ensures the original context menu doesn't proceed
}

// Extension Definition
app.registerExtension({
    name: "efficiency.seedcontrol",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (NODE_WIDGET_MAP[nodeData.name]) {
            addMenuHandler(nodeType, function (insertOption) {
                // Check conditions before showing the seed behavior option
                let showSeedOption = true;

                if (nodeData.name === "Noise Control Script") {
                    // Check for 'add_seed_noise' widget being false
                    const addSeedNoiseWidget = this.widgets.find(w => w.name === 'add_seed_noise');
                    if (addSeedNoiseWidget && !addSeedNoiseWidget.value) {
                        showSeedOption = false;
                    }
                } else if (nodeData.name === "HighRes-Fix Script") {
                    // Check for 'use_same_seed' widget being true
                    const useSameSeedWidget = this.widgets.find(w => w.name === 'use_same_seed');
                    if (useSameSeedWidget && useSameSeedWidget.value) {
                        showSeedOption = false;
                    }
                }

                if (showSeedOption) {
                    insertOption({
                        content: "🌱 Seed behavior...",
                        has_submenu: true,
                        callback: showSeedBehaviorMenu
                    });
                }
            });
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
                this.seedControl = new SeedControl(this, NODE_WIDGET_MAP[nodeData.name]);
                this.seedControl.seedWidget.value = -1;
            };
        }
    },
});
