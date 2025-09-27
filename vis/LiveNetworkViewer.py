import pygame
import math
import random
from neat.genome import Genome


class NetworkVisualizer:
    def __init__(self, width=1200, height=800):
        self.width = width
        self.height = height
        self.screen = None
        self.num_file = 1
        # Colors
        self.bg_color = (20, 20, 30)
        self.node_color = (100, 150, 255)
        self.input_color = (100, 255, 100)
        self.output_color = (255, 100, 100)
        self.hidden_color = (255, 255, 100)
        self.connection_color = (200, 200, 200)
        self.disabled_color = (100, 100, 100)

    def visualize_network(self, genome, title="Evolved Neural Network", filename="network"):
        """Render a genome's neural network once and save as image"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title)
        font = pygame.font.SysFont("Arial", 14)
        title_font = pygame.font.SysFont("Arial", 24, bold=True)

        self.screen.fill(self.bg_color)

        # Build network structure
        network_info = self.analyze_network(genome)
        node_positions = self.calculate_node_positions(network_info)

        # Draw title
        title_text = title_font.render(title, True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(self.width // 2, 30))
        self.screen.blit(title_text, title_rect)

        # Draw network info
        info_y = 60
        info_texts = [
            f"Nodes: {len(network_info['nodes'])} | Connections: {len(genome.genes)}",
            f"Input: {network_info['input_count']} | Hidden: {network_info['hidden_count']} | Output: {network_info['output_count']}",
            f"Fitness: {genome.fitness:.2f}",
        ]
        for text in info_texts:
            text_surface = font.render(text, True, (200, 200, 200))
            self.screen.blit(text_surface, (10, info_y))
            info_y += 20

        # Draw network
        self.draw_connections(genome, node_positions, font)
        self.draw_nodes(network_info, node_positions, font)
        self.draw_legend(font)

        # Save screenshot
        pygame.image.save(self.screen, f"./visualization-snapshots/{filename}_{self.num_file:03d}.png")
        print(f"Network visualization saved as {filename}_{self.num_file:03d}.png")
        self.num_file += 1
        pygame.quit()

    def analyze_network(self, genome):
        """Analyze network structure"""
        nodes = {}
        input_count = 0
        output_count = 0
        hidden_count = 0

        # Collect all node numbers from genes
        all_node_nums = set()
        for gene in genome.genes:
            in_id = gene.in_node.number
            out_id = gene.out_node.number
            all_node_nums.add(in_id)
            all_node_nums.add(out_id)

        # Categorize nodes
        for node_num in all_node_nums:
            if node_num < genome.gh.n_inputs:
                node_type = "input"
                input_count += 1
            elif node_num < genome.gh.n_inputs + genome.gh.n_outputs:
                node_type = "output"
                output_count += 1
            else:
                node_type = "hidden"
                hidden_count += 1

            nodes[node_num] = {
                'type': node_type,
                'number': node_num,
                'connections_in': [],
                'connections_out': []
            }

        # Add connection information
        for gene in genome.genes:
            if gene.enabled:
                in_id = gene.in_node.number
                out_id = gene.out_node.number
                nodes[in_id]['connections_out'].append(gene)
                nodes[out_id]['connections_in'].append(gene)

        return {
            'nodes': nodes,
            'input_count': input_count,
            'output_count': output_count,
            'hidden_count': hidden_count
        }

    def calculate_node_positions(self, network_info):
        """Calculate positions for all nodes"""
        positions = {}
        margin = 100

        # Separate nodes by type
        input_nodes = [n for n in network_info['nodes'].values() if n['type'] == 'input']
        output_nodes = [n for n in network_info['nodes'].values() if n['type'] == 'output']
        hidden_nodes = [n for n in network_info['nodes'].values() if n['type'] == 'hidden']

        # Position input nodes (left side)
        if input_nodes:
            input_x = margin
            input_spacing = (self.height - 2 * margin) / max(1, len(input_nodes) - 1) if len(input_nodes) > 1 else 0
            for i, node in enumerate(input_nodes):
                y = margin + i * input_spacing if len(input_nodes) > 1 else self.height // 2
                positions[node['number']] = (input_x, int(y))

        # Position output nodes (right side)
        if output_nodes:
            output_x = self.width - margin
            output_spacing = (self.height - 2 * margin) / max(1, len(output_nodes) - 1) if len(output_nodes) > 1 else 0
            for i, node in enumerate(output_nodes):
                y = margin + i * output_spacing if len(output_nodes) > 1 else self.height // 2
                positions[node['number']] = (output_x, int(y))

        # Position hidden nodes (middle, arranged in layers)
        if hidden_nodes:
            # Simple layout: arrange hidden nodes in the middle
            hidden_layers = 3  # Assume max 3 hidden layers for layout
            layer_width = (self.width - 2 * margin - 200) / (hidden_layers + 1)

            nodes_per_layer = len(hidden_nodes) // hidden_layers + 1

            for i, node in enumerate(hidden_nodes):
                layer = i // nodes_per_layer
                pos_in_layer = i % nodes_per_layer

                x = margin + 100 + layer * layer_width

                if nodes_per_layer > 1:
                    y_spacing = (self.height - 2 * margin) / (nodes_per_layer - 1)
                    y = margin + pos_in_layer * y_spacing
                else:
                    y = self.height // 2

                positions[node['number']] = (int(x), int(y))

        return positions

    def draw_connections(self, genome, positions, font):
        """Draw all connections between nodes"""
        for gene in genome.genes:
            in_id = gene.in_node.number
            out_id = gene.out_node.number

            if in_id in positions and out_id in positions:
                start_pos = positions[in_id]
                end_pos = positions[out_id]

                # Color based on weight and enabled status
                if gene.enabled:
                    # Color intensity based on weight strength
                    weight_strength = min(abs(gene.weight), 3.0) / 3.0
                    if gene.weight > 0:
                        color = (int(100 + 155 * weight_strength), int(50 + 50 * weight_strength), 50)  # Red
                    else:
                        color = (50, int(50 + 50 * weight_strength), int(100 + 155 * weight_strength))  # Blue

                    # Line thickness based on weight
                    thickness = max(1, int(weight_strength * 5))
                else:
                    color = self.disabled_color
                    thickness = 1

                pygame.draw.line(self.screen, color, start_pos, end_pos, thickness)

                # Draw weight value near the middle of the connection
                if gene.enabled and abs(gene.weight) > 0.1:  # Only show significant weights
                    mid_x = (start_pos[0] + end_pos[0]) // 2
                    mid_y = (start_pos[1] + end_pos[1]) // 2

                    weight_text = font.render(f"{gene.weight:.1f}", True, (255, 255, 255))
                    text_rect = weight_text.get_rect(center=(mid_x, mid_y))

                    # Draw background for text
                    pygame.draw.rect(self.screen, (0, 0, 0, 128), text_rect.inflate(4, 2))
                    self.screen.blit(weight_text, text_rect)

    def draw_nodes(self, network_info, positions, font):
        """Draw all nodes"""
        for node_num, pos in positions.items():
            node_info = network_info['nodes'][node_num]

            # Choose color based on node type
            if node_info['type'] == 'input':
                color = self.input_color
                radius = 15
            elif node_info['type'] == 'output':
                color = self.output_color
                radius = 15
            else:  # hidden
                color = self.hidden_color
                radius = 12

            # Draw node circle
            pygame.draw.circle(self.screen, (0, 0, 0), pos, radius + 2)  # Border
            pygame.draw.circle(self.screen, color, pos, radius)

            # Draw node number
            text = font.render(str(node_num), True, (0, 0, 0))
            text_rect = text.get_rect(center=pos)
            self.screen.blit(text, text_rect)

            # Draw node label below
            if node_info['type'] == 'input':
                if node_num == 0:
                    label = "Radar 0°"
                elif node_num == 1:
                    label = "Radar 45°"
                elif node_num == 2:
                    label = "Radar 90°"
                elif node_num == 3:
                    label = "Radar 135°"
                elif node_num == 4:
                    label = "Radar 180°"
                elif node_num == 5:
                    label = "Speed"
                else:
                    label = f"Input {node_num}"
            elif node_info['type'] == 'output':
                if node_num == 6:
                    label = "Steering"
                elif node_num == 7:
                    label = "Accelerate"
                elif node_num == 8:
                    label = "Brake"
                else:
                    label = f"Output {node_num}"
            else:
                label = f"Hidden {node_num}"

            label_text = font.render(label, True, (200, 200, 200))
            label_rect = label_text.get_rect(center=(pos[0], pos[1] + radius + 15))
            self.screen.blit(label_text, label_rect)

    def draw_legend(self, font):
        """Draw legend explaining the visualization with transparency"""
        legend_x = self.width - 250
        legend_y = self.height - 150

        # Create a transparent surface for the legend background
        legend_surface = pygame.Surface((240, 140), pygame.SRCALPHA)

        # Draw semi-transparent background (RGBA: Red, Green, Blue, Alpha)
        # Alpha values: 0 = fully transparent, 255 = fully opaque
        pygame.draw.rect(legend_surface, (40, 40, 60, 180), (0, 0, 240, 140))  # Semi-transparent dark background
        pygame.draw.rect(legend_surface, (100, 100, 120, 200), (0, 0, 240, 140), 2)  # Semi-transparent border

        # Blit the transparent background to the main screen
        self.screen.blit(legend_surface, (legend_x - 10, legend_y - 10))

        legend_items = [
            ("Legend:", (255, 255, 255, 220)),  # Slightly transparent white
            ("🟢 Input Nodes", (*self.input_color, 220)),  # Add transparency to existing colors
            ("🔴 Output Nodes", (*self.output_color, 220)),
            ("🟡 Hidden Nodes", (*self.hidden_color, 220)),
            ("Red Lines: Positive weights", (255, 100, 100, 220)),
            ("Blue Lines: Negative weights", (100, 100, 255, 220)),
            ("Thickness = Weight strength", (200, 200, 200, 220))
        ]

        # Create individual text surfaces with per-pixel alpha for text transparency
        for i, (text, color) in enumerate(legend_items):
            # Create a surface for the text with per-pixel alpha
            text_surface = font.render(text, True, color[:3])  # Use RGB only for font rendering

            # If we want text transparency, create a new surface with alpha
            if len(color) == 4:  # If alpha value is provided
                alpha_surface = pygame.Surface(text_surface.get_size(), pygame.SRCALPHA)
                alpha_surface.fill((255, 255, 255, color[3]))  # Fill with white and desired alpha
                text_surface.blit(alpha_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

            self.screen.blit(text_surface, (legend_x, legend_y + i * 18))

    def draw_network_once(self, genome, title="Network"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(title)
            self.font = pygame.font.SysFont("Arial", 14)
            self.title_font = pygame.font.SysFont("Arial", 24, bold=True)

        # Handle quit events so window is responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.screen.fill(self.bg_color)

        network_info = self.analyze_network(genome)
        node_positions = self.calculate_node_positions(network_info)

        # Draw title
        title_text = self.title_font.render(title, True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(self.width // 2, 30))
        self.screen.blit(title_text, title_rect)

        # Draw connections and nodes
        self.draw_connections(genome, node_positions, self.font)
        self.draw_nodes(network_info, node_positions, self.font)
        self.draw_legend(self.font)

        pygame.display.flip()


def visualize_best_network(population, generation=None):
    """Visualize the best network from a population and save to file"""
    best_genome = max(population.population, key=lambda g: g.fitness)
    visualizer = NetworkVisualizer()
    filename = f"best_network_gen_{generation}.png" if generation is not None else "best_network.png"
    visualizer.visualize_network(best_genome, f"Best Network (Fitness: {best_genome.fitness:.2f})", filename)



# Usage example - add this to your main code after training:
if __name__ == "__main__":
    # After your training loop, add:
    visualize_best_network(pop)