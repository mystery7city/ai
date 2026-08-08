#include <iostream>
#include <string>

class Character
{
private:
    std::string name;
    int hp;

public:
    Character(const std::string& character_name, int character_hp)
        : name{character_name}, hp{character_hp}
    {
    }

    void introduce() const
    {
        std::cout << "Name: " << name << '\n';
        std::cout << "HP: " << hp << '\n';
    }

    void take_damage(int damage)
    {
        if (damage < 0)
        {
            return;
        }

        hp -= damage;

        if (hp < 0)
        {
            hp = 0;
        }
    }

    bool is_alive() const
    {
        return hp > 0;
    }
};

int main()
{
    Character hero{"Hero", 100};

    hero.introduce();

    std::cout << "\nHero takes 35 damage.\n";
    hero.take_damage(35);
    hero.introduce();

    std::cout << "\nIs hero alive? "
              << (hero.is_alive() ? "Yes" : "No")
              << '\n';

    return 0;
}